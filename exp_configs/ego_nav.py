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
MAX_PROMPT_LEN=600

n_envs = 2
total_timesteps = 2_000_000
save_freq = 20_000

# what type of vec env to use
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
vec_env_cls = SubprocVecEnv
MAX_SIM_STEPS=600
# PPO hyperparameters
ppo = dict(
    n_steps=128,
    batch_size=8,
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
    use_policy_head=False,
    policy_token_idx=-1,
    value_token_idx=-1,
    lora_config =  LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
                task_type="CAUSAL_LM",
    )
)

# Callback to run on the trainer prior to training starts
def modify_ppo(model:PPO):
    model._update_learning_rate = lambda x:None #Hack to disable the destructive LR updates
    param_groups = [
            {"params": model.policy.lvlm.parameters(), "lr": 1e-6},  # frozen or tiny lr
            {"params": model.policy.value_net.parameters(), "lr": 3e-4},   # **higher LR**
            # {"params": model.policy.policy_net.parameters(),"lr": 3e-4}
    ]
    import torch
    base_lr = model.learning_rate
    eps = model.policy.optimizer.defaults.get("eps", 1e-5)
    betas = model.policy.optimizer.defaults.get("betas", (0.9, 0.999))
    weight_decay = model.policy.optimizer.defaults.get("weight_decay", 0.001)

    model.policy.optimizer = torch.optim.Adam(
        param_groups,
        lr=base_lr,         # SB3 still uses this as "reference" LR, but overridden by group LRs
        eps=eps,
        betas=betas,
        weight_decay=weight_decay
    )

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Extra callbacks
callbacks = []

category_mapping = {'chair': 0, 'bed': 1, 'plant': 2, 'toilet': 3, 'tv_monitor': 4, 'sofa': 5}
id_to_name_mapping = {v: k for k, v in category_mapping.items()}

# How we turn observations into prompts!
def prompt_formatter(obs):
    from PIL import Image
    target_object = id_to_name_mapping[obs['objectgoal'][0]]
    chat = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Your goal is to find the {target_object}. Navigate to it and stop."},
            {"type": "image"},
            {"type": "text", "text": "Choose: 0=stop 1=move_forward 2=turn_left 3=turn_right"}]},
        {"role": "assistant", "content": ""}
    ]
    return chat, [Image.fromarray(obs['rgb'])]

def historical_prompt_formatter(obs):
    from PIL import Image
    target_object = id_to_name_mapping[obs['objectgoal'][0]]
    frame_times = obs['frame_times']
    chat = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Your goal is to find the {target_object}. Navigate to it and stop."},
            {"type":"text","text":f"observation from {frame_times[0]}"},
            {"type": "image"},
            {"type":"text","text":f"observation from {frame_times[1]}"},
            {"type": "image"},
            {"type":"text","text":f"observation from {frame_times[2]}"},
            {"type": "image"},
            {"type":"text","text":f"observation from {frame_times[3]}"},
            {"type": "image"},
            {"type": "text", "text": "Choose: 0=stop 1=move_forward 2=turn_left 3=turn_right"}]},
        {"role": "assistant", "content": ""}
    ]
    return chat, [Image.fromarray(image) for image in obs['rgb']]

def make_env():
    # --- Standard Habitat Imports ---
    from habitat.config.default import get_config
    from habitat import make_dataset
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        TopDownMapMeasurementConfig,
        FogOfWarConfig,
    )
    from habitat.gym import make_gym_from_config
    
    # --- Compatibility Imports ---
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
    from envs.egocentric_nav import HabitatRenderWrapper
    import numpy as np
    
    # --- THIS IS THE FIX ---
    # Import both libraries with distinct names
    import gym as old_gym  # The library Habitat-gym uses
    # import gymnasium       # The library SB3/Shimmy uses (no 'as gym')
    # -------------------------

    # --- Your Config Logic ---
    config_env = get_config("configs/""objectnav_hm3d_rgbd_semantic.yaml")
    with read_write(config_env):
        config_env.habitat.dataset.split = "val"
        config_env.habitat.task.measurements.top_down_map = (
            TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=512,
                draw_goal_positions=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_border=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5, fov=72),
            )
        )
    dataset = make_dataset(
        config_env.habitat.dataset.type, config=config_env.habitat.dataset
    )
    
    # 1. Create the old env
    env = make_gym_from_config(config_env, dataset)

    # 2. Apply the patch
    for key, space in env.observation_space.spaces.items():
        
        # --- THIS IS THE FIX ---
        # Check using the OLD gym library
        if isinstance(space, old_gym.spaces.Box):
        # -------------------------
        
            if (space.low > space.high).any():
                print(f"   WARNING: Fixing invalid observation space for '{key}'.")
                print(f"     Old space: low={space.low.min()}, high={space.high.max()}")

                new_high = np.maximum(space.low, space.high)

                # --- THIS IS THE FIX ---
                # Create a new space using the OLD gym library
                new_space = old_gym.spaces.Box(
                    low=space.low,
                    high=new_high,
                    shape=space.shape,
                    dtype=space.dtype,
                )
                # -------------------------

                env.observation_space.spaces[key] = new_space
                print(f"     New space: low={new_space.low.min()}, high={new_space.high.max()}")
    shimmy_env = GymV21CompatibilityV0(env=env)
    
    # 4. (NEW) Apply the render wrapper
    final_env = HabitatRenderWrapper(shimmy_env)

    return final_env


ENV_WRAPPER = partial(
    lambda env: LLMProcessingWrapper(
        env,
        processor=processor,
        prompt_formatter=prompt_formatter,
        max_length=MAX_PROMPT_LEN
    )
)
