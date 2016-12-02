# universe-starter-agent

The codebase implements a starter agent that can solve a number of `universe` environments.
It contains a basic implementation of the [A3C algorithm](https://https://arxiv.org/abs/1602.01783), adapted for real-time environments.

# Dependencies

* Python 2.7 or 3.5
* six (for py2/3 compatibility)
* TensorFlow 0.11
* tmux (the start script opens up a tmux session with multiple windows)
* gym
* universe
* opencv-python
* numpy
* scipy

# Getting Started

## Atari Pong

`python train.py --num-workers 2 --env-id PongDeterministic-v3 --log-dir /tmp/pong`

The command above will train an agent on Atari Pong using ALE simulator.
It will see two workers that will be learning in parallel (`--num-workers` flag) and will output intermediate results into given directory.

The code will launch the following processes:
* worker-0 - a process that runs policy gradient
* worker-1 - a process identical to process-1, that uses different random noise from the environment
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning

Once you start the training process, it will create a tmux session with a window for each of these processes. You can connect to them by typing `tmux a` in the console.
To see a window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To access TensorBoard to see various monitoring metrics of the agent, type: `http://localhost:22012/` in chrome.

Using 16 workers, the agent should be able to solve Pong within 30 minutes on an `m4.10xlarge` instance.
Using 32 workers, the agent is abel to solve Pong in 10 minutes on an `m4.16xlarge` instance.
If you run this experiment on a high-end macbook pro, the above code will take just under 2 hours to solve Pong.

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/tb_pong.png "Pong")

For best performance, it is recommended for the number of workers to not exceed available number of CPU cores.

You can stop the experiment with `tmux kill-session` command.

## Playing games over remote desktop

The main difference with the previous experiment is that now we are going to play the game through VNC protocol.
The VNC environments are hosted on the EC2 cloud and have an interface that's different from a conventional Atari Gym
environment;  luckily, with the help of several wrappers (which are used within `envs.py` file)
the experience should be similar to the agent as if it was played locally. The problem itself is more difficult
because the observations and actions are delayed due to the latency induced by the network.

More interestingly, you can also peek at what the agent is doing with a VNCViewer.

### Atari

`python train.py --num-workers 2 --env-id gym-core.PongDeterministic-v3 --log-dir /tmp/vncpong`

_Peeking into the agent's environment with TurboVNC_

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/vnc_pong.png "Pong over VNC")

#### Important caveats

One of the novel challenges in using Universe environments is that
they operate in *real time*, and in addition, it takes time for the
environment to transmit the observation to the agent.  This time
creates a lag: where the greater the lag, the harder it is to solve
environment with today's RL algorithms.  Thus, to get the best
possible results it is necessary to reduce the lag, which can be
achieved by having both the environments/allocator and the agent live
on the same high-speed computer network.  So for example, if you have
a fast local network, you could host the environments on one set of
machines, and the agent on another machine that can speak to the
environments with low latency.  Alternatively, you can run the
environments and the agent on the same EC2/Azure region.  Other configurations
tend to have greater lag.

To keep track of your lag, look for the phrase `reaciton_time` in stdout.  If you run both the agent
and the environment on nearby machines on the cloud, your `reaction_time` should be as low as 40ms.
The `reaction_time` statistic is printed to stderr because we wrap our environment with the `Logger` wrapper,
as done in [here](<https://github.com/openai/universe-starter-agent/blob/master/envs.py#L32>).

Generally speaking, environments that are most affected by lag are
games that place a lot of emphasis on reaction time.  For example, this
agent is able to solve Universe Pong (`gym-core.PongDeterministic-v3`)
in under 2 hours when both the agent and the environment
are co-located on the cloud, but this agent had difficulty solving
Universe Pong when the environment was on the cloud while the agent
was not.  This issue affects environments that place great emphasis on
reaction time.

### Playing flash games

You may run the following command to launch the agent on the game Neon Race:

`python train.py --num-workers 2 --env-id flashgames.NeonRace-v0 --log-dir /tmp/duskdrive`

_What agent sees when playing Neon Race_
![dusk](https://github.com/openai/universe-starter-agent/raw/master/imgs/neon_race.png "Neon Race")



### Next steps

Now that you've seen how to interact with universe
