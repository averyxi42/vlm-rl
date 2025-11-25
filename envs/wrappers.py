import gymnasium as gym
class FrameSkip(gym.Wrapper):
    """
    A wrapper that repeats the same action for a specified number of frames
    and returns the observation from the final frame.

    :param env: The environment to wrap.
    :param skip: The number of frames to skip for each action.
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip <= 0:
            raise ValueError("Frame skip must be a positive integer.")
        self.skip = skip

    def step(self, action):
        """
        Repeat the action `skip` times, returning the AVERAGE reward.
        """
        total_reward = 0.0
        steps_taken = 0
        obs, reward, terminated, truncated, info = None,None,None,None,None
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps_taken += 1
            
            # If the episode ends mid-skip, we must stop and return immediately
            if terminated or truncated:
                break
        
        # The crucial change: divide the total reward by the number of steps taken.
        average_reward = total_reward / steps_taken
        
        return obs, average_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment."""
        return self.env.reset(**kwargs)
