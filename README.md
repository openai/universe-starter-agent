# universe-starter-agent

The codebase implements a starter agent that can interact with `universe`.
It contains a basic implementation of the A3C algorithm, adapted for real-time environments.

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
It will se two workers that will be learning in parallel (`--num-workers` flag) and will output intermediate results into given directory.

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

### Playing flash games

`python train.py --num-workers 2 --env-id flashgames.DuskDrive-v0 --log-dir /tmp/duskdrive`

_What agent sees when playing Dusk Drive_
![dusk](https://github.com/openai/universe-starter-agent/raw/master/imgs/dusk_drive.png "Dusk Drive")


### Next steps

Now that you've seen how to interact with universe
