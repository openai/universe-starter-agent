import argparse
import os

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--num-workers', default=1, type=int, help="Number of workers")
parser.add_argument("--env-id", type=str, default="VNCPongDeterministic-v3", help="Environment id")
parser.add_argument("--log-dir", type=str, default="/tmp/pong", help="Log directory path")

def new_tmux_cmd(name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {} '{}' Enter".format(name, cmd)

def create_tmux_commands(session, num_workers, env_id, logdir):
    base_cmd = ['python', 'worker.py', '--log-dir', logdir, '--env-id', env_id, '--num-workers', str(num_workers)]

    cmds_map = [new_tmux_cmd("ps", base_cmd + ["--job-name", "ps"])]
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd(
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i)])]

    cmds_map += [new_tmux_cmd("tb", ["tensorboard --logdir {} --port 12012".format(logdir)])]
    cmds_map += [new_tmux_cmd("htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(logdir),
        "tmux kill-session",
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

    cmds = create_tmux_commands("a3c", args.num_workers, args.env_id, args.log_dir)
    print("\n".join(cmds))
    os.system("\n".join(cmds))


if __name__ == "__main__":
    run()
