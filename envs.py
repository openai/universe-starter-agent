import cv2
from gym.spaces.box import Box
import numpy as np
import gym
import logging
from universe import vectorized
from universe.wrappers import BlockingReset, DiscreteToVNCAction, EpisodeID, Unvectorize
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_env(env_id, n=1, **kwargs):
    spec = gym.spec(env_id)
    remotes = "http://allocator.sci.openai-tech.com?n={}".format(n)

    if spec.tags.get('flashgames', False):
        return create_flash_env(env_id, remotes, **kwargs)
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return create_vncatari_env(env_id, remotes, **kwargs)

def create_flash_env(env_id, remotes, **_):
    raise NotImplementedError()

def _process_frame42(frame):
    frame = frame[:210, :160]
    frame = cv2.resize(frame, (84, 110))
    frame = frame.mean(2)
    frame = frame[18:102, :]
    frame = cv2.resize(frame, (42, 42))
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0, 255, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]

def create_vncatari_env(env_id, remotes, **_):
    env = gym.make(env_id)
    assert env.metadata['runtime.vectorized']
    env = BlockingReset(env)
    env = DiscreteToVNCAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    # env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps)
    return env
