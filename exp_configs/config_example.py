# configs/experiment_car_racing.py

from envs.car_racing import HumanRGBRacingEnv
from envs.wrappers import FrameSkip
from vlm_policies import LVLMActorCriticPolicy
from transformers import AutoProcessor
from functools import partial
from peft import LoraConfig
from stable_baselines3 import PPO
from vlm_policies import LLMProcessingWrapper
# basic parameters
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PROMPT_LEN=560

n_envs = 6
total_timesteps = 2_000_000
save_freq = 20_000

# what type of vec env to use
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
vec_env_cls = SubprocVecEnv

# PPO hyperparameters
ppo = dict(
    n_steps=256,
    batch_size=6,
    n_epochs=1,
    gamma=0.9,
    ent_coef=0.0,
    max_grad_norm=1.0,
    # learning_rate=3e-6,
    clip_range=0.2,
    clip_range_vf=0.2,
    normalize_advantage=False
)

# Policy configuration
policy = dict(
    model_name=MODEL_NAME,
    exploration_temperature=1.0,
    detach_value_head=True,
    use_policy_head=True,
    policy_token_idx=-1,
    value_token_idx=-1,
    lora_config =  LoraConfig(
                r=128,
                lora_alpha=256,
                lora_dropout=0.05,
                target_modules=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
                task_type="CAUSAL_LM",
    )
)

# Callback to run on the trainer prior to training starts
def modify_ppo(model:PPO):
    model._update_learning_rate = lambda:None #Hack to disable the destructive LR updates
    model.policy.optimizer.param_groups = [
            {"params": model.policy.lvlm.parameters(), "lr": 1e-6},  # frozen or tiny lr
            {"params": model.policy.value_net.parameters(), "lr": 3e-4},   # **higher LR**
            {"params": model.policy.policy_net.parameters(),"lr": 3e-4}
    ]

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Extra callbacks
callbacks = []

# How we turn observations into prompts!
def prompt_formatter(obs):
    from PIL import Image
    chat = [
        {"role": "user", "content": [
            {"type": "text", "text": "Drive well."},
            {"type": "image"},
            {"type": "text", "text": "Choose: 0=Coast 1=Right 2=Left 3=Gas"}
        ]},
        {"role": "assistant", "content": ""}
    ]
    return chat, [Image.fromarray(obs)]

def make_env():
    return FrameSkip(HumanRGBRacingEnv(), skip=5)

ENV_WRAPPER = partial(
    lambda env: LLMProcessingWrapper(
        env,
        processor=processor,
        prompt_formatter=prompt_formatter,
        max_length=MAX_PROMPT_LEN
    )
)


# def car_racing_formatter(obs):
#     from PIL import Image

#     """
#     A prompt formatter for the CarRacing-v3 environment.
    
#     It discretizes the continuous action space and creates a detailed prompt.
#     The continuous action space is Box(-1..1, (3,)), for steer, gas, brake.
#     We will create 5 discrete actions:
#     0: Steer Left
#     1: Steer Right
#     2: Gas
#     3: Brake
#     4: Straight (No action)
#     """
    
#     # prompt_text = (
#     #     "You are an expert driver in a race. Your goal is to drive the car "
#     #     "around the track (grey road) as fast as possible without going off-road (green grass). The image "
#     #     "is a birds eye view of your car, with the white number on the lower left being your speed in MPH.\n\n"
#     #     "Choose one of the following discrete actions:\n"
#     #     "[action id 1]: Steer left\n"
#     #     "[action id 2]: Steer right\n"
#     #     "[action id 3]: Accelerate\n"
#     #     "[action id 0:] Coast (do nothing)\n\n"
#     #     "if your speed is low <10MPH, choose id 3 to accelerate.\n"
#     #     "however, if your speed is above 20MPH, prioritize steering to stay on the road by choosing id 1,0,or 2\n"
#     #     "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting."
#     # )
#     prompt_text = (
#         "You are an expert driver in a race. Your goal is to drive the car "
#         "around the track (grey road) as fast as possible without going off-road (green grass). The image "
#         "is a birds eye view of your car, with the white bar on the lower left indicating your speed.\n\n"
#         "Choose one of the following discrete actions:\n"
#         "[action id 1]: Steer Right\n"
#         "[action id 2]: Steer Left\n"
#         "[action id 3]: Accelerate\n"
#         "[action id 0:] Coast (do nothing)\n\n"
#         "note: if you see no white bar on the lower left, your are stationary and you must choose 3 to accelerate.\n"
#         "however, if your speed high, prioritize steering to stay on the road by choosing id 1,0,or 2\n"
#         "you MUST choose by selecting a single number corresponding to the desired action id, with NO extra formatting."
#     )
#     # The modern, correct chat format
#     chat = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Analyze the racetrack from this image:"},
#                 {"type": "image"}, # Image placeholder
#                 {"type": "text", "text": prompt_text},
#             ]
#         }
#         ,{"role": "assistant", "content": ''},

#     ]
    
#     # The processor expects a list of PIL Images
#     images = [Image.fromarray(obs)]
    
#     return chat, images