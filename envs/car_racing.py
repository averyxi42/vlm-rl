import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2 # Using OpenCV for resizing
from typing import Tuple,List,Dict
from PIL import Image
class HumanRGBRacingEnv(gym.Env):
    """
    A version of the Taxi-v3 environment where the observation is the
    human-readable 'rgb_array' render, resized for deep learning.
    This presents a more realistic perception challenge for the agent.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode='rgb_array',img_shape = [600,400, 3],max_steps=1000):
        # We need the internal environment to be in 'rgb_array' mode
        self.internal_env = gym.make("CarRacing-v3", render_mode='rgb_array',continuous=False,max_episode_steps=max_steps)
        self.render_mode = render_mode
        self.img_shape = img_shape
        # The observation space is a standard 84x84 RGB image
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        self.action_space = self.internal_env.action_space

        self.window = None
        self.clock = None

    def _get_obs(self):
        # Get the RGB array from the underlying environment
        frame = self.internal_env.render()
        # Resize to a standard dimension for CNNs
        resized_frame = cv2.resize(frame, self.img_shape[:2], interpolation=cv2.INTER_AREA)
        return resized_frame

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # The internal env's reset doesn't return an observation, so we call render
        self.internal_env.reset(seed=seed, options=options)
        return self._get_obs(), {}

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.internal_env.step(action)
        self.step_count+=1
        if(self.step_count>1000):
            print("error! step count exceeded!")
            exit()
        if(truncated):
            print(f"truncated after {self.step_count} base steps")
        if terminated:
            print(f"terminated after {self.step_count} base steps")
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # This render method is for human viewing during inference/evaluation
        if self.render_mode == "human":
            # Get the original, high-res frame for display
            frame = self.internal_env.render()
            if self.window is None:
                import pygame
                pygame.init()
                self.window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                pygame.display.set_caption("HumanRGBTaxiEnv")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else: # "rgb_array"
            return self._get_obs()

    def close(self):
        self.internal_env.close()
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
