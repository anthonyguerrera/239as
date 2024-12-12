import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from utils import preprocess #this is a helper function that may be useful to grayscale and crop the image


class EnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env:gym.Env,
        skip_frames:int=4,
        stack_frames:int=4,
        initial_no_op:int=50,
        do_nothing_action:int=0,
        **kwargs
    ):
        """the environment wrapper for CarRacing-v2

        Args:
            env (gym.Env): the original environment
            skip_frames (int, optional): the number of frames to skip, in other words we will
            repeat the same action for `skip_frames` steps. Defaults to 4.
            stack_frames (int, optional): the number of frames to stack, we stack 
            `stack_frames` frames to form the state and allow agent understand the motion of the car. Defaults to 4.
            initial_no_op (int, optional): the initial number of no-op steps to do nothing at the beginning of the episode. Defaults to 50.
            do_nothing_action (int, optional): the action index for doing nothing. Defaults to 0, which should be correct unless you have modified the 
            discretization of the action space.
        """
        super(EnvWrapper, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.do_nothing_action = do_nothing_action
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(stack_frames, 84, 84),
            dtype=np.float32
        )
        
        
    # Implmenented reset/step following discussion video
    
    def reset(self, **kwargs):
        # call the environment reset
        state, info = self.env.reset(**kwargs)
        
        # Do nothing for the next self.initial_no_op' steps
        for _ in range(self.initial_no_op):
            state, reward, terminated, truncated, info = self.env.step(self.do_nothing_action)
        
        # Crop and resize the frame
        state = preprocess(state)
        
        # stack the frames to form the initial state
        self.stacked_state = np.tile(state, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    
    # Call with every action to move the environment forward
    def step(self, action):
        total_reward = 0
        
        # For all of the skip_frames (generally 4) take the same action, since car doesn't move much in one frame
        for _ in range(self.skip_frames):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
            
        # shift all the states up by 1 and put in new state at the end
        state = preprocess(state)
        self.stacked_state = np.concatenate((self.stacked_state[1:], state[np.newaxis]), axis=0)
            
        
            
        return self.stacked_state, total_reward, terminated, truncated, info
        
    
    
   