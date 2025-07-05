import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation


def make_env(env_name, stack_size):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84,84))
    env = FrameStack(env, stack_size)
    return env
    