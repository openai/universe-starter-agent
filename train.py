import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                    help="Log directory path")


def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def create_tmux_commands(session, num_workers, remotes, env_id, logdir):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
        '--log-dir', logdir, '--env-id', env_id,
        '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    cmds_map = [new_tmux_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]])]

    cmds_map += [new_tmux_cmd(session, "tb", ["tensorboard --logdir {} --port 12345".format(logdir)])]
    cmds_map += [new_tmux_cmd(session, "htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(logdir),
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d".format(session, windows[0]),
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {}".format(session, w)]
    cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


def run():
    args = parser.parse_args()

    cmds = create_tmux_commands("a3c", args.num_workers, args.remotes, args.env_id, args.log_dir)
    print("\n".join(cmds))
    os.system("\n".join(cmds))

if __name__ == "__main__":
    run()
