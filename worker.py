#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    # TODO(rafal): Create model and the trainer here!

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
    init_op = tf.initialize_variables(variables_to_save)
    init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')
    summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)
    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             # TODO(rafal): Fix!
                             # global_step=trainer.global_network.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        trainer.start(sv, summary_writer, sess)
        global_step = sess.run(network.global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(network.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--cluster', required=True, help='JSON dump of the cluster spec to use')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--remotes', help='What VNC remotes to use')
    args = parser.parse_args()
    spec = json.loads(args.cluster)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    if args.job_name == "worker":
        logger.info('Running with: cluster_spec=%s remotes=%s', spec, args.remotes)
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        logger.info('Cluster spec: %s', spec)
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))

        server.join()

if __name__ == "__main__":
    tf.app.run()
