# universe-starter-agent

The codebase implements a starter agent that can solve a number of `universe` environments.
It contains a basic implementation of the `A3C algorithm <https://https://arxiv.org/abs/1602.01783>_`, adapted for real-time environments.

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
* worker-0 - the first process that learns to solve the game
* worker-1 - the second process that learns to solve the game
* ps - process that synchronizes parameters of the agent across different workers (parameter server)
* tb - tensorboard for monitoring progress of the agent

Once you start the training, it will create a tmux session with a window for each of these processes. You can connect to them by typing `tmux a` in console.
To see a window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To see various monitoring metrics of the agent, type: `http://localhost:22012/` in chrome, which will open TensorBoard.

_Using 16 workers, the agent should be able to solve Pong within 30 minutes on an `m4.10xlarge` instance_
_Using 32 workers, the agent is abel to solve Pong in 10 minutes on an `m4.16xlarge` instance_.
If you run this experiment on a high-end macbook pro, the above code will take just under 2 hours to solve Pong.

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/tb_pong.png "Pong")

For best performance, it is recommended for the number of workers to not exceed available number of CPU cores.

You can stop the experiment with `tmux kill-session` command.

## Playing games over remote desktop

The main difference to the previous experiment is that we are now going to play the game through VNC protocol.
Environments are hosted on EC2 cloud but with the help of various wrappers (take a look at `envs.py` file)
the experience should be similar to the agent as if it was played locally. The problem itself is more complicated
because observations and actions are delayed due to network latencies.

More interestingly, you can also peek what the agent is doing with a VNCViewer.

### Atari

`python train.py --num-workers 2 --env-id gym-core.PongDeterministic-v3 --log-dir /tmp/vncpong`

_Peeking into the agent's environment with TurboVNC_

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/vnc_pong.png "Pong over VNC")

#### Important caveats

One of the key challenges in using universe environments is that they operate in *real time*, and that in addition,
it takes time for the environment process to transmit the observation pixels to the agent.  This time creates a lag:
where the greater the lag, the harder it is to solve environment with today's RL algorithms.  While the existence of
the lag creates a challenge, an additional challenge is created by the fact that the lag depends on the speed of the
network.  Thus, to get the best possible results, we strongly recommend that both the environments/allocator and the agent
live on the same network.  So for example, if you have a fast local network, you could place the environments on one set
of machines, and the agent on another machine that  can speak to the environments very quickly.  Alternatively, you can
run the environments and the agent on the same EC2/Azure region.  If you do not do it, and, for example, run the environment
on the cloud while the agent is run on your local machine, then the lag will be larger and performance will be worse.

In general, environments that are most affected by lag are games that are primarily focused on reaction time.  For example,
this agent is able to solve Universe Pong (gym-core.PongDeterministic-v3)
reasonably in under 2 hours when both the agent and the environment are co-located
on the cloud, but this agent had difficulty solving Universe Pong when the environment was on the cloud while the
agent was not.  This issue affects environments that place great emphasis on reaction time.  

### Playing flash games

`python train.py --num-workers 2 --env-id flashgames.DuskDrive-v0 --log-dir /tmp/duskdrive`

_What agent sees when playing Dusk Drive_
![dusk](https://github.com/openai/universe-starter-agent/raw/master/imgs/dusk_drive.png "Dusk Drive")


### Next steps

Now that you've seen how to interact with universe
