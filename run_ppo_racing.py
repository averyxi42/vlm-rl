# train_taxi_lvlm.py

import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from transformers import AutoProcessor
import gymnasium as gym
# --- 1. Import our custom components from their files ---
from envs.taxi import HumanRGBTaxiEnv,OracleHumanRGBTaxiEnv,taxi_formatter,cheater_taxi_formatter

# --- (Assume these are in your project structure as well) ---
from vlm_policies import LLMProcessingWrapper, GenerativeLVLMPolicy, MacroActionWrapper
from visualization.visualization_utils import SimpleRenderCallback

print("--- All components imported successfully ---")


# --- 2. Initialize the LVLM Processor ---
# MODEL_NAME='google/gemma-3-4b-it'
# MODEL_NAME='Qwen/Qwen3-VL-4B-Instruct'
MODEL_NAME='Qwen/Qwen2.5-VL-3B-Instruct'
# --- 4. Configuration for the PPO Agent ---
# For a complex model like an LVLM, we need a large rollout buffer to get
# stable gradients.
N_STEPS = 6#396       # Number of steps to collect per rollout
BATCH_SIZE = 6      # Mini-batch size for each gradient update
N_EPOCHS = 1        # How many times to loop over the collected data
LEARNING_RATE = 3e-5 # A smaller learning rate is often better for fine-tuning
DEVICE = 'cuda:1' 
LOGDIR = './ppo_car_qwen_test/'
N_ENVS = 2 # Number of parallel environments.


processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"--- Processor for {MODEL_NAME} loaded ---")
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
from envs.car_racing import HumanRGBRacingEnv
# --- 3. Create the Vectorized Environment ---
base_env_creator = lambda:FrameSkip(HumanRGBRacingEnv(),skip=5)
from PIL import Image
def car_racing_formatter(obs):
    """
    A prompt formatter for the CarRacing-v2 environment.
    
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
    #     "[action id 1]: Steer right\n"
    #     "[action id 2]: Steer left\n"
    #     "[action id 3]: Accelerate\n"
    #     "[action id 0:] Coast (do nothing)\n\n"
    #     "if your speed is low <10MPH, choose id 3 to accelerate.\n"
    #     "however, if your speed is above 20MPH, prioritize steering to stay on the road by choosing id 1,0,or 2\n"
    #     "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting."
    # )
    prompt_text = (
        "You are an expert driver in a simulated race. Your goal is to drive the car "
        "around the track (colored GREY) as fast as possible without going off-road (colored GREEN). The image "
        "is a birds eye view of your car, with the white bar on the lower left indicating your true speed.\n\n"
        "Choose one of the following discrete actions:\n"
        "[action id 1]: Steer right\n"
        "[action id 2]: Steer left\n"
        "[action id 3]: Accelerate\n"
        "[action id 0:] Coast (do nothing)\n\n"
        "if you see no white bar or minimal white bar on the lower left next to the numbers, your are stationary and you must choose **3** to accelerate.\n"
        # "trying to steer when you are stationary has no effect and results in a negative reward.\n"
        # "if the map appears zoomed out, the game has just started and you should acclerate by choosing "
        # "choose by selecting a single number corresponding to the desired action id."
        "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting or explanations."

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

formatter = car_racing_formatter
# We use a dictionary to pass arguments to our custom wrapper.
# This ensures each of the parallel environments is correctly wrapped.
llm_wrapper_kwargs = {
    'processor': processor,
    'prompt_formatter': formatter,
    'max_length': 560
}

print("\n--- Creating a temporary test environment to inspect the prompt... ---")

# a. Create a single instance of the base environment
test_base_env = base_env_creator()

# b. Wrap it with the LLM processor, same as the vec_env will be
test_wrapped_env = LLMProcessingWrapper(
    test_base_env,
    processor=processor,
    prompt_formatter=formatter,
    max_length=llm_wrapper_kwargs['max_length']  # Use the same max_length as your final env
)

# c. Reset the environment to get a sample observation
print("Getting a sample observation...")
sample_obs, _ = test_wrapped_env.reset()

print('input ids:')
print(sample_obs['input_ids'])
# d. Print the crucial information
# input_ids_length = sample_obs['input_ids'].shape[1]
# print("\n--- Observation Sample ---")
# print("Keys:", sample_obs.keys())
# print(f"Tokenized 'input_ids' length: {input_ids_length}")
# print(f"Max length parameter: {test_wrapped_env.max_length}")

# if input_ids_length >= test_wrapped_env.max_length:
#     print("\n\033[93mWARNING: The tokenized prompt length is equal to or greater than max_length.")
#     print("Your prompt is likely being truncated! Consider increasing max_length.\033[0m")
# else:
#     print("\033[92mSUCCESS: The prompt fits within the max_length.\033[0m")

# e. Clean up the temporary environment
test_wrapped_env.close()
print("\n--- Test environment closed. Proceeding to create VecEnv for training... ---")
# Use SB3's make_vec_env. Since we are not using a registered Gym ID,
# we pass the class itself. We also do not need env_kwargs.
# The LLMProcessingWrapper is applied to each parallel environment.

MAX_MACRO_ACTION_LEN = 6
NO_ACTION_PENALTY = -0.03
MULTI_ACTION_PENALTY = -0.03

# --- 2. Prepare Keyword Arguments for EACH Wrapper ---

# Kwargs for the INNER wrapper: MacroActionWrapper
macro_wrapper_kwargs = {
    "tokenizer": processor.tokenizer,
    "max_macro_action_len": MAX_MACRO_ACTION_LEN,
    "no_action_penalty": NO_ACTION_PENALTY,
    "multi_action_penalty": MULTI_ACTION_PENALTY
}

# --- 3. Use a Lambda Function to Chain the Wrappers ---

print("--- Creating vectorized environment with chained wrappers... ---")

vec_env = make_vec_env(
    base_env_creator,
    n_envs=N_ENVS,
    # The 'wrapper_class' is now a lambda that defines the full chain.
    # The 'env' variable is the base environment instance provided by make_vec_env.
    wrapper_class=lambda env: LLMProcessingWrapper(
        MacroActionWrapper(env, **macro_wrapper_kwargs), # This is the inner wrapper
        **llm_wrapper_kwargs                          # This is the outer wrapper
    )
    # Note: `wrapper_kwargs` is no longer used, as we pass args directly in the lambda.
)



# vec_env = make_vec_env(
#     base_env_creator,
#     n_envs=N_ENVS,
#     wrapper_class=LLMProcessingWrapper,
#     wrapper_kwargs=llm_wrapper_kwargs
# )

print(f"\n--- VecEnv created with {vec_env.num_envs} parallel environments. ---")
print("--- All components are ready. Preparing PPO agent... ---")

os.makedirs(LOGDIR, exist_ok=True)



torch.cuda.empty_cache()

# --- 5. Instantiate the PPO Agent ---
# We use the standard PPO class from Stable Baselines 3.
# The magic is in passing our custom LVLMActorCriticPolicy.
render_callback = SimpleRenderCallback(render_freq=1, image_path=LOGDIR)
vec_env.render_mode = 'rgb_array'

# NOTE: Adjust the device ('cuda:0', 'cuda:1', etc.) to match your hardware setup.

model = PPO(
    policy=GenerativeLVLMPolicy,
    env=vec_env,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    verbose=1,
    tensorboard_log=LOGDIR,
    policy_kwargs={ "exploration_temperature": 1.0,'model_name':MODEL_NAME,"max_new_tokens":MAX_MACRO_ACTION_LEN,"processor_kwargs":{"max_pixels":28*28*1024}},
    clip_range_vf=0.2, # Clip the value function update
    clip_range=0.2,    # Standard clipping for the policy
    max_grad_norm=1,
    normalize_advantage=False, # Advantage normalization can be problematic.
    ent_coef=0.001,
    gamma=0.94,
    device=DEVICE
)

print("\n--- PPO Agent Instantiated Successfully! ---")
print(f"Policy Class: {type(model.policy)}")
print(f"Rollout buffer size: {model.n_steps} steps")
print(f"Device: {model.device}")


# --- 6. Start the Training Loop ---
# This is the final step. The `learn()` method orchestrates the
# entire process of collecting rollouts and training the LVLM policy.
print("\n==================================================")
print("             STARTING PPO TRAINING              ")
print("==================================================")
try:
    # Let's train for a set number of timesteps
    model.learn(total_timesteps=200000, callback=render_callback,progress_bar=True)
    
    print("\n==================================================")
    print("           TRAINING COMPLETED SUCCESSFULLY        ")
    print("==================================================")

    # Save the final model for later use
    model.save(LOGDIR+"ppo_lvlm_taxi")
    print("\nModel saved to ppo_lvlm_taxi.zip")

except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("             TRAINING FAILED!                     ")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Error:", e)
    import traceback
    traceback.print_exc()

finally:
    # It's crucial to close the vectorized environment.
    vec_env.close()