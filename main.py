import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    
    done = True

    for step in range(5000):
        if done: state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        print(state.shape)
        env.render()

    env.close()


if __name__ == "__main__":
    main()