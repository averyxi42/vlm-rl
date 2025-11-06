# ppo_example.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv # <-- IMPORTANT CHANGE HERE
from stable_baselines3.common.callbacks import BaseCallback
import imageio
import threading
import os
from monitor_app import run_app
import functools
class RenderCallback(BaseCallback):
    """
    Custom callback for rendering the agent's behavior in a dedicated environment.
    This avoids interfering with the vectorized training environments.
    """
    def __init__(self, render_env_id: str, render_freq: int, verbose: int = 0):
        super(RenderCallback, self).__init__(verbose)
        self.render_env_id = render_env_id
        self.render_freq = render_freq
        # These will be initialized in _on_training_start
        self.render_env = None
        self.obs = None

    def _on_training_start(self) -> None:
        """Create the rendering environment before training starts."""
        self.render_env = gym.make(self.render_env_id, render_mode="rgb_array")
        # Gymnasium's reset() returns a tuple (observation, info)
        self.obs, _ = self.render_env.reset()

    def _on_step(self) -> bool:
        """Render the environment at a specified frequency."""
        if self.n_calls % self.render_freq == 0:
            # Get the current policy to predict the next action
            action, _ = self.model.predict(self.obs, deterministic=True)
            
            # Step the rendering environment
            self.obs, _, terminated, truncated, _ = self.render_env.step(action)
            
            # Render the current frame to an RGB array and save it
            img = self.render_env.render()
            print("image:")
            print(img)
            imageio.imwrite('render.png', img)

            # If the episode in the render environment is over, reset it
            if terminated or truncated:
                self.obs, _ = self.render_env.reset()
        return True
    
    def _on_training_end(self) -> None:
        """Close the rendering environment after training is finished."""
        if self.render_env is not None:
            self.render_env.close()

from visualization_utils import SimpleRenderCallback

# This guard is crucial for multiprocessing and good practice in general
if __name__ == '__main__':
    # --- Configuration ---
    ENV_ID = 'CartPole-v1'
    # For DummyVecEnv, NUM_CPU is the number of environments run sequentially
    NUM_CPU = 4  
    TOTAL_TIMESTEPS = 30000
    # Render the environment every N steps.
    # A higher value means less frequent updates but lower overhead.
    RENDER_FREQ = 30

    # --- Start Web Server in a Background Thread ---
    # print("Starting monitoring web server at http://localhost:5000")
    # flask_thread = threading.Thread(target=run_app, daemon=True)
    # flask_thread.start()

    # --- Create the Vectorized Training Environment ---
    # Using DummyVecEnv for stability. It runs envs sequentially in the same process.
    train_env = make_vec_env(ENV_ID, n_envs=NUM_CPU, env_kwargs={'render_mode': 'rgb_array'})
    print(train_env.render_mode)
    train_env.render_mode='rgb_array'
    # train_env.render()
    # train_env = gym.make(ENV_ID, render_mode='rgb')
    # env_fns = [lambda: gym.make(ENV_ID, render_mode='rgb_array') for _ in range(NUM_CPU)]

    # We then pass this list of functions directly to the VecEnv constructor.
    # This gives us full control and bypasses SB3's problematic automatic wrapping.
    # train_env = DummyVecEnv(env_fns)
    # env_fns = [lambda: gym.make(ENV_ID, render_mode='rgb_array') for _ in range(NUM_CPU)]
    # def make_env_with_render(env_id, render_mode):
    #     # Explicitly create the base environment with the render_mode
    #     base_env = gym.make(env_id, render_mode=render_mode)
    #     # Return this properly initialized environment
    #     return base_env

    # env_fns = [
    #     functools.partial(make_env_with_render, ENV_ID, 'rgb_array')
    #     for _ in range(NUM_CPU)
    # ]
    # # We then pass this list of functions directly to the VecEnv constructor.
    # train_env = DummyVecEnv(env_fns)
    # --- Create the PPO Model ---
    model = PPO('MlpPolicy', train_env, verbose=1)

    # --- Create the Custom Monitoring Callback ---
    render_callback = SimpleRenderCallback(render_freq=RENDER_FREQ,image_path='render_output')

    # --- Train the Model ---
    print("\nStarting PPO training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=render_callback)

    # --- Clean Up ---
    train_env.close()
    print("\nTraining finished.")
    print("You can now close the web browser and stop the script with CTRL+C.")