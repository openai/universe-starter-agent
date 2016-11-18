from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import queue
import scipy.signal
import threading

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_td = discount(delta_t, gamma * lambda_)

    features = {}
    keys = rollout.features[0].keys()

    for key in keys:
        feature_values = []
        for i in range(len(rollout.features)):
            assert key in rollout.features[i]
            feature_values.append(rollout.features[i][key])
        features[key] = feature_values

    return Batch(batch_si, batch_a, batch_td, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "td", "r", "terminal", "features"])

class PartialRollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    def __init__(self, env, policy, num_local_steps, timeout=600.0):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.run_queue = queue.Queue(1)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_state = None
        self.last_features = None
        self.logger = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.started = False
        self.timeout = timeout

    def start_runner(self, sess):
        self.sess = sess
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps)
        while True:
            self.queue.put(next(rollout_provider), timeout=self.timeout)

def env_runner(env, policy, num_local_steps):
    last_state = env.reset()
    last_features = policy.reset()
    length = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            action, value_, features = policy.sample_policy_and_value(last_state, last_features)
            action_for_env = policy.postprocess_action(action)
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action_for_env.argmax())

            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1

            last_state = state
            last_features = features

            if terminal or length >= env.spec.timestep_limit:
                terminal_end = True
                if length >= env.spec.timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.reset()
                length = 0
                break

        if not terminal_end:
            rollout.r, _ = policy.run_value(last_state, last_features)
        yield rollout

class A3C(object):
    def __init__(self, env, task):
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            self.opt_step = tf.get_variable("opt_step", [], tf.int32, initializer=tf.zeros_initializer, trainable=False)
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                               trainable=False)

        with tf.device(worker_device):
            self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)

        self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)

        pi_loss = - tf.reduce_sum(tf.reduce_sum(prob_tf * self.ac, [1]) * self.adv)
        vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
        entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

        bs = tf.to_float(tf.shape(pi.x)[0])
        tf.scalar_summary("model/policy_loss", pi_loss / bs)
        tf.scalar_summary("model/value_loss", vf_loss / bs)
        tf.scalar_summary("model/entropy", entropy / bs)
        self.loss = pi_loss + vf_loss - entropy * 0.01
        self.runner = RunnerThread(env, pi, 20)

        grads = tf.gradients(self.loss, pi.var_list)
        self.summary_op = tf.merge_all_summaries()
        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

        grads_and_vars = list(zip(grads, self.network.var_list))
        inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

        opt = tf.train.AdamOptimizer(1e-4)
        self.train_op = tf.group(opt.apply_gradients(grads_and_vars, self.opt_step), inc_step)

    def pull_batch_from_queue(self):
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        global_step = sess.run(self.global_step)

        should_compute_summary = self.task == 0 and global_step % 101 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op]
        else:
            fetches = [self.train_op]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.td,
            self.r: batch.r,
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            pass
            # TODO(rafal): add tensorboard
            # self.logger.log(global_step, tf.Summary.FromString(fetched[0]))
