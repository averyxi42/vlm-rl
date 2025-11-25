# %%
# =================================================================================
# Cell 1: Setup and Environment Creation
# =================================================================================
# This cell installs necessary packages, creates a standard image-based
# environment, and defines a prompt_formatter for it.

# --- 1. Install Dependencies ---
# Note: You might need to restart the kernel after this installation.

import gymnasium as gym
from PIL import Image
import numpy as np
from typing import Any, Tuple, List, Dict
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.car_racing import HumanRGBRacingEnv
from envs.wrappers import FrameSkip

if __name__=="__main__":
    print("--- Dependencies Installed ---")

    # --- 2. Create the Standard Environment ---
    # CarRacing-v3 is a great choice because its observation is a standard
    # (96, 96, 3) RGB image array, perfect for our LVLM.
    # env = gym.make("CarRacing-v3")
    env = HumanRGBRacingEnv()
    print("\n--- CarRacing-v3 Environment Created ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    # The action space is continuous, but we will discretize it in the formatter.

    # --- 3. Define the Corresponding Prompt Formatter ---
    # This function is tailored specifically for the CarRacing-v3 environment.
    def car_racing_formatter(obs: np.ndarray) -> Tuple[List[Dict], List[Image.Image]]:
        """
        A prompt formatter for the CarRacing-v3 environment.
        
        It discretizes the continuous action space and creates a detailed prompt.
        The continuous action space is Box(-1..1, (3,)), for steer, gas, brake.
        We will create 5 discrete actions:
        0: Steer Left
        1: Steer Right
        2: Gas
        3: Brake
        4: Straight (No action)
        """
        
        # prompt_text = (
        #     "You are an expert driver in a race. Your goal is to drive the car "
        #     "around the track (grey road) as fast as possible without going off-road (green grass). The image "
        #     "is a birds eye view of your car, with the white number on the lower left being your speed in MPH.\n\n"
        #     "Choose one of the following discrete actions:\n"
        #     "[action id 1]: Steer left\n"
        #     "[action id 2]: Steer right\n"
        #     "[action id 3]: Accelerate\n"
        #     "[action id 0:] Coast (do nothing)\n\n"
        #     "if your speed is low <10MPH, choose id 3 to accelerate.\n"
        #     "however, if your speed is above 20MPH, prioritize steering to stay on the road by choosing id 1,0,or 2\n"
        #     "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting."
        # )
        prompt_text = (
            "You are an expert driver in a race. Your goal is to drive the car "
            "around the track (grey road) as fast as possible without going off-road (green grass). The image "
            "is a birds eye view of your car, with the white bar on the lower left indicating your speed.\n\n"
            "Choose one of the following discrete actions:\n"
            "[action id 1]: Steer Right\n"
            "[action id 2]: Steer Left\n"
            "[action id 3]: Accelerate\n"
            "[action id 0:] Coast (do nothing)\n\n"
            "note: if you see no white bar on the lower left, your are stationary and you must choose 3 to accelerate.\n"
            "however, if your speed high, prioritize steering to stay on the road by choosing id 1,0,or 2\n"
            "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting."
        )
        # The modern, correct chat format
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the racetrack from this image:"},
                    {"type": "image"}, # Image placeholder
                    {"type": "text", "text": prompt_text},
                ]
            }
            ,{"role": "assistant", "content": ''},

        ]
        
        # The processor expects a list of PIL Images
        images = [Image.fromarray(obs)]
        
        return chat, images

    print("\n--- `car_racing_formatter` Defined ---")

    # --- 4. Test the Formatter with a Sample Observation ---
    print("\n--- Testing the formatter with one sample... ---")
    sample_obs, _ = env.reset()
    sample_chat, sample_images = car_racing_formatter(sample_obs)

    print("Formatter produced a chat structure:")
    import json
    print(json.dumps(sample_chat, indent=2))
    print(f"\nFormatter produced {len(sample_images)} image(s).")
    print(f"Image type: {type(sample_images[0])}, Image size: {sample_images[0].size}")

    # Close the environment to free up resources
    env.close()

    print("\n--- Cell execution finished. You can now use `car_racing_formatter` ---")
    print("--- and the environment name 'CarRacing-v3' to test the wrapper. ---")


    # %%
    # =================================================================================
    # Cell 6: Advanced Environment with Frame Stacking via Prompt Formatting
    # =================================================================================
    # This cell defines a new, stateful prompt formatter that handles frame stacking
    # internally, creating a history of captioned images for the LVLM.

    import gymnasium as gym
    from PIL import Image
    import numpy as np
    from typing import Any, Tuple, List, Dict
    from collections import deque
    from vlm_policies import LLMProcessingWrapper,LVLMActorCriticPolicy
    from transformers import AutoProcessor
    # --- Ensure previous components exist ---
    # MODEL_NAME='google/gemma-3-4b-it'
    MODEL_NAME='Qwen/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    assert 'LLMProcessingWrapper' in locals(), "Please run the cell defining LLMProcessingWrapper."
    assert 'processor' in locals(), "Please run a cell that defines the `processor`."

    print("--- Defining Advanced Frame Stacking Formatter ---")


    # --- 5. Create the Final, Fully-Equipped Environment ---
    print("\n--- Creating the final, wrapped environment with frame stacking... ---")

    # a. Create the base environment
    # base_env = gym.make("CarRacing-v3",render_mode="rgb_array",continuous=False)
    base_env = HumanRGBRacingEnv()
    base_env = FrameSkip(base_env,skip=5)
    # b. Apply the action discretizer
    # discretized_env = DiscretizedCarRacing(base_env)

    # c. Apply the LLM Processing wrapper with our new stateful formatter
    env = LLMProcessingWrapper(
        base_env,
        processor=processor, 
        prompt_formatter=car_racing_formatter, # Use the new formatter object
        max_length=560, # Increase max_length for the larger prompt
    )

    print("\n--- Environment `env` is now ready for your manual rollout cell! ---")
    print("It provides a history of 4 images in the prompt at each step.")

    # --- Quick Test ---
    print("\n--- Testing one step of the new environment... ---")
    obs, _ = env.reset()
    print(obs['input_ids'])
    print("Observation received. Keys:", obs.keys())
    print("Shape of 'input_ids':", obs['input_ids'].shape) # Should have a larger sequence length
    # env.close()

    # %%
    # =================================================================================
    # Cell 7: End-to-End PPO Training Test
    # =================================================================================
    # This cell instantiates the SB3 PPO agent with our custom components
    # and launches the training loop.

    import torch
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # --- 1. Ensure Environment and Policy are Ready ---
    assert 'env' in locals(), "Please run the previous cell to create the `env` object."
    # assert 'policy' in locals(), "Please run the cell that instantiates the `policy` object."

    # Since the policy is already instantiated, we don't need to pass `policy_kwargs`.
    # We pass the instantiated policy object directly.
    # However, the action space of the policy must match the environment's. Let's verify.
    # assert policy.action_space == env.action_space, \
    #     f"Action space mismatch! Policy has {policy.action_space} but Env has {env.action_space}"
    # --- 2. Create the Vectorized Environment ---
    N_ENVS = 6 # Number of parallel environments. 8 is a good starting point.
    # base_env_id = "CarRacing-v3"

    # We use a partial function to pass arguments to our custom wrapper.
    # This ensures each of the 8 parallel environments is correctly wrapped.
    wrapper_kwargs = {
        'processor': processor,
        'prompt_formatter': car_racing_formatter,
        'max_length': 560
    }
    # We also apply the FrameSkip wrapper to each environment
    # env_kwargs = {'continuous': False}
    # base_env_creator = HumanRGBRacingEnv# lambda:FrameSkip(HumanRGBRacingEnv(),skip=5)
    base_env_creator = lambda:LLMProcessingWrapper(FrameSkip(HumanRGBRacingEnv(), skip=5),**wrapper_kwargs)
    # Use SB3's make_vec_env to handle the creation of parallel processes
    vec_env = make_vec_env(
        base_env_creator,
        n_envs=N_ENVS,
        # env_kwargs=env_kwargs,
        # This is a bit complex: we want to wrap with FrameSkip THEN LLMProcessingWrapper
        # A cleaner way is to define a single function that does the wrapping.
        # wrapper_class=lambda env: LLMProcessingWrapper(FrameSkip(env, skip=5), **wrapper_kwargs),
        vec_env_cls=SubprocVecEnv

    )

    print(f"\n--- VecEnv created with {vec_env.num_envs} parallel environments. ---")
    print("--- All components are ready. Preparing PPO agent... ---")

    LOGDIR = './ppo_g0_9_n6_l256/'
    # --- 2. Configuration for the PPO Agent ---
    # For a complex model like an LVLM, we need a large rollout buffer to get
    # stable gradients. We also want to keep it on the CPU to save VRAM.
    N_STEPS = 256#2048# 512       # Number of steps to collect per rollout
    BATCH_SIZE = 6     # Mini-batch size for each gradient update max is 6
    N_EPOCHS = 1        # How many times to loop over the collected data
    LEARNING_RATE = 3e-6 # A smaller learning rate is often better for fine-tuning
    DEVICE='cuda:1'
    DISABLE_LR_SCHEDULE = True
    torch.cuda.empty_cache()
    # --- 3. Instantiate the PPO Agent ---
    # We use the standard PPO class from Stable Baselines 3.
    # The magic is in passing our custom, pre-instantiated policy object.
    from visualization.visualization_utils import SimpleRenderCallback
    render_callback = SimpleRenderCallback(render_freq=1,image_path=LOGDIR)  #600,400, 1176,1176
    vec_env.render_mode='rgb_array'
    model = PPO(
        policy=LVLMActorCriticPolicy,
        env=vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        verbose=1,
        tensorboard_log=LOGDIR,
        policy_kwargs={"exploration_temperature":1.0,'model_name':MODEL_NAME},
        clip_range_vf=0.2, # Clip the value function update
        clip_range=0.2,    # Standard clipping for the policy
        max_grad_norm=1,
        normalize_advantage=False, #latest change; advantage normalization is problematic.
        ent_coef=0.00,
        gamma=0.9,
        device=DEVICE,
        # learning_rate=5e-6,
        # verbose=1
    )

    print("\n--- PPO Agent Instantiated Successfully! ---")
    print(f"Policy Class: {type(model.policy)}")
    print(f"Rollout buffer size: {model.n_steps} steps")
    print(f"Device: {model.device}")


    # --- 4. Start the Training Loop ---
    # This is the final step. The `learn()` method will now orchestrate the
    # entire process: collecting rollouts by calling `env.step()`, and then
    # training the policy by calling its `evaluate_actions()` method.
    print("\n==================================================")
    print("             STARTING PPO TRAINING              ")
    print("==================================================")
    try:
        # Let's train for a set number of timesteps to see if it works
        model.learn(total_timesteps=200000,callback=render_callback,progress_bar=True)
        
        print("\n==================================================")
        print("           TRAINING COMPLETED SUCCESSFULLY        ")
        print("==================================================")

        # Save the final model for later use
        model.save(LOGDIR+"ppo_lvlm_car_racing")
        print("\nModel saved to ppo_lvlm_car_racing.zip")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("             TRAINING FAILED!                     ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error:", e)
        import traceback
        traceback.print_exc()

    finally:
        # It's crucial to close the environment to shut down renderers etc.
        env.close()