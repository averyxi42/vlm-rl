import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv

# from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
if __name__=="__main__":
    # --- 1. Environment Setup ---
    # Environment ID
    ENV_ID = "CarRacing-v3"
    # Number of parallel environments to run
    N_ENVS = 16
    # Number of frames to stack
    N_STACK = 4
    LOGDIR= "./ppo_carracing_baseline/"
    # Create the vectorized environment
    # The lambda function is a common way to properly initialize parallel environments
    # The continuous=True argument is the default for CarRacing-v2
    # env = make_vec_env(lambda: gym.make(ENV_ID, continuous=True), n_envs=N_ENVS,)
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=N_ENVS, # Use the same N_ENVS as your setup
        env_kwargs={"render_mode": "rgb_array", "continuous": True},
        vec_env_cls=SubprocVecEnv
    )

    from visualization.visualization_utils import SimpleRenderCallback
    render_callback = SimpleRenderCallback(render_freq=5,image_path=LOGDIR)  #600,400, 1176,1176
    env.render_mode='rgb_array'

    # --- 2. Apply Wrappers ---
    # Grayscale the observations
    # env = GrayScaleObservation(env, keep_dim=True)
    # # Resize the observations to a smaller size
    # env = ResizeObservation(env, 64)
    # Stack N_STACK frames
    env = VecFrameStack(env, n_stack=N_STACK)
    # env.render_mode='rgb_array'

    # Important: Stable Baselines 3 expects channel-first images (C, H, W) for CNN policies
    # The environment originally provides (H, W, C), so we need to transpose it
    env = VecTransposeImage(env)

    # env.render_mode='rgb_array'

    # --- 3. PPO Model Configuration ---
    # Hyperparameters inspired by SB3 RL Zoo and common PPO tuning
    ppo_params = {
        "policy": "CnnPolicy",
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2, # A common value for PPO clipping
        "ent_coef": 0.0,   # No entropy bonus
        "verbose": 1,
        "tensorboard_log": LOGDIR
    }

    # Create the PPO model
    model = PPO(env=env, **ppo_params)

    # --- 4. Training ---
    # Total timesteps to train for. CarRacing requires a lot of training.
    # Start with a smaller number like 1e5 to test, then increase to 1e6 or more.
    TOTAL_TIMESTEPS = 1_000_000

    # Add a progress bar to the training
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True,callback=render_callback)

    # --- 5. Save and Close ---
    model.save("ppo_carracing_model")
    env.close()

    print("Training finished and model saved!")