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

n_envs = 2
total_timesteps = 2_000_000
save_freq = 20_000

# what type of vec env to use
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
vec_env_cls = SubprocVecEnv
MAX_SIM_STEPS=1000
# PPO hyperparameters
ppo = dict(
    n_steps=512,
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
            {"params": model.policy.policy_net.parameters(),"lr": 3e-4}
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
    return FrameSkip(HumanRGBRacingEnv(max_steps=MAX_SIM_STEPS), skip=5)

ENV_WRAPPER = partial(
    lambda env: LLMProcessingWrapper(
        env,
        processor=processor,
        prompt_formatter=prompt_formatter,
        max_length=MAX_PROMPT_LEN
    )
)

# The purpose of this config is to test the usage of both action and policy heads instead of the default LM head,
# Ideally isolating the 
