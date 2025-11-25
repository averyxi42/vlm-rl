from stable_baselines3.common.callbacks import BaseCallback
import imageio
from pathlib import Path
import os
class SimpleRenderCallback(BaseCallback):
    """
    A simple callback to render the first environment in the VecEnv and save the image.
    This is the efficient and correct way to monitor training.
    """
    def __init__(self, render_freq: int, verbose: int = 0,image_path='.',image_name='current.png'):
        super(SimpleRenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.image_path = Path(image_path)
        self.image_name = image_name
    def _on_step(self) -> bool:
        """
        This method is called after each step in the training process.
        """
        # Render the environment every `render_freq` steps.
        if self.n_calls % self.render_freq == 0:
            # self.training_env is the VecEnv object used for training.
            # .render() automatically targets the first environment in the vector.
            img = self.training_env.render()
            # Save the image. We use the step number for a unique filename.
            # imageio.imwrite(f'renders/step_{self.n_calls}.png', img)
            os.makedirs(self.image_path,exist_ok=True)
            imageio.imwrite(str(self.image_path / self.image_name), img)
        return True
    
class DiscountedReturnCallback(BaseCallback):
    """
    A custom callback to track the discounted return of each episode
    and log it to TensorBoard.
    """
    def __init__(self, verbose: int = 0):
        super(DiscountedReturnCallback, self).__init__(verbose)
        self.discounted_rewards = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        Used to initialize the running discounted rewards.
        """
        # Initialize a running discounted reward for each parallel environment
        self.discounted_rewards = np.zeros(self.model.n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        """
        This method is called after each step in the training process.
        """
        # Get the rewards and dones from the step
        # self.locals['rewards'] is a numpy array of rewards for each env
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        
        # Update the running discounted reward for each environment
        # The formula is: G_t = R_t+1 + gamma * G_t+1
        # In our case, this translates to:
        # new_discounted_reward = current_reward + gamma * previous_discounted_reward
        self.discounted_rewards = rewards + self.model.gamma * self.discounted_rewards

        for i, done in enumerate(dones):
            if done:
                # An episode has ended for this environment
                final_discounted_return = self.discounted_rewards[i]
                
                # Log the value to TensorBoard
                self.logger.record("rollout/discounted_ep_rew_mean", final_discounted_return)
                
                # Reset the running total for this environment for the next episode
                self.discounted_rewards[i] = 0.0
                
        return True