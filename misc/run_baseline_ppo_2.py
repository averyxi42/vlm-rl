import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv

# --- (This is the only assumed dependency from your project) ---
from visualization.visualization_utils import SimpleRenderCallback

print("--- All components imported successfully ---")

# ==============================================================================
#  1. SELF-CONTAINED FRAMESKIP WRAPPER
# ==============================================================================
if __name__ == "__main__":
    class FrameSkip(gym.Wrapper):
        """
        A wrapper that repeats the same action for a specified number of frames.
        (Identical to the one in your LVLM script for an apples-to-apples comparison)
        """
        def __init__(self, env: gym.Env, skip: int = 4):
            super().__init__(env)
            if skip <= 0:
                raise ValueError("Frame skip must be a positive integer.")
            self.skip = skip

        def step(self, action):
            total_reward = 0.0
            steps_taken = 0
            for _ in range(self.skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps_taken += 1
                if terminated or truncated:
                    break
            average_reward = total_reward / steps_taken
            return obs, average_reward, terminated, truncated, info

    # ==============================================================================
    #  2. CONFIGURATION MIRRORED FROM LVLM SCRIPT
    # ==============================================================================

    # --- Training Hyperparameters ---
    N_STEPS = 256             # Number of steps to collect per rollout
    BATCH_SIZE = 32          # Mini-batch size for each gradient update
    N_EPOCHS = 4            # How many times to loop over the collected data
    LEARNING_RATE = 3e-5    # Learning rate
    N_ENVS = 64              # Number of parallel environments
    FRAME_SKIP = 5          # Frame skip value
    N_STACK = 4             # Number of frames to stack for the CNN

    # --- PPO Algorithm Hyperparameters ---
    GAMMA = 0.94
    ENT_COEF = 0.001
    CLIP_RANGE = 0.2
    CLIP_RANGE_VF = 0.2
    MAX_GRAD_NORM = 1.0
    NORMALIZE_ADVANTAGE = False

    # --- Setup Configuration ---
    DEVICE = 'cuda:2'       # Match the device from your script
    LOGDIR = './runs/ppo_car_cnn_baseline_subproc_50/' # A separate log directory for the baseline

    # ==============================================================================
    #  3. ENVIRONMENT SETUP
    # ==============================================================================

    print("--- Creating vectorized environment for the CNN baseline... ---")

    # Define a lambda function that creates a single environment instance.
    # This is the standard way to set up VecEnv with wrappers.
    env_creator = lambda: FrameSkip(
        gym.make("CarRacing-v3", continuous=False,render_mode='rgb_array'),
        skip=FRAME_SKIP
    )

    # Create the vectorized environment
    vec_env = make_vec_env(env_creator, n_envs=N_ENVS,vec_env_cls=SubprocVecEnv)
    vec_env.render_mode='rgb_array'
    # Apply standard wrappers for image-based RL with CNNs
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    vec_env = VecTransposeImage(vec_env) # Pytorch expects (C, H, W)

    print(f"\n--- VecEnv created with {vec_env.num_envs} parallel environments. ---")
    print(f"Observation Space (after wrappers): {vec_env.observation_space.shape}")
    print(f"Action Space (Discrete): {vec_env.action_space.n} actions")
    print("Action Meanings: [0: Do Nothing, 1: Steer Left, 2: Steer Right, 3: Gas, 4: Brake]")

    # ==============================================================================
    #  4. MODEL INSTANTIATION AND TRAINING
    # ==============================================================================

    os.makedirs(LOGDIR, exist_ok=True)
    render_callback = SimpleRenderCallback(render_freq=3, image_path=LOGDIR)

    model = PPO(
        policy='CnnPolicy',
        env=vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        clip_range_vf=CLIP_RANGE_VF,
        max_grad_norm=MAX_GRAD_NORM,
        normalize_advantage=NORMALIZE_ADVANTAGE,
        verbose=1,
        tensorboard_log=LOGDIR,
        device=DEVICE
    )

    print("\n--- PPO Agent (CnnPolicy) Instantiated Successfully! ---")
    print(f"Policy Class: {type(model.policy)}")
    print(f"Rollout buffer size (n_steps * n_envs): {model.n_steps * model.n_envs}")
    print(f"Device: {model.device}")

    print("\n==================================================")
    print("        STARTING PPO CNN BASELINE TRAINING        ")
    print("==================================================")
    try:
        model.learn(total_timesteps=200000, callback=render_callback, progress_bar=True)
        
        print("\n==================================================")
        print("           TRAINING COMPLETED SUCCESSFULLY        ")
        print("==================================================")

        model.save(LOGDIR + "ppo_cnn_carracing_baseline")
        print(f"\nModel saved to {LOGDIR}ppo_cnn_carracing_baseline.zip")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("             TRAINING FAILED!                     ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error:", e)
        import traceback
        traceback.print_exc()

    finally:
        # Ensure the environment is closed properly
        vec_env.close()