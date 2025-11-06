import json
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from transformers import AutoProcessor

# By importing babyai and minigrid, register the environments with gymnasium
import babyai
import minigrid

from vlm_policies import LLMProcessingWrapper, LVLMActorCriticPolicy
from visualization.visualization_utils import SimpleRenderCallback


# =================================================================================
# 1: Constants & Configuration
# =================================================================================
MODEL_NAME = "google/gemma-3-4b-it"
ENV_ID = "BabyAI-GoToObjMazeS7-v0"
MAX_STEPS_PER_EPISODE = 120
TOKENIZER_MAX_LENGTH = 512
LOGDIR = "./ppo_lvlm_babyai/"


DIRECTION_STRINGS: Dict[int, str] = {
    0: "facing east",
    1: "facing south",
    2: "facing west",
    3: "facing north",
}

ACTION_MAP: Dict[int, str] = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up an object",
    4: "drop",
    5: "toggle",
    6: "done",
}


# =================================================================================
# 2: Prompt Formatter
# =================================================================================
def babyai_prompt_formatter(obs: Dict[str, Any]) -> Tuple[List[Dict], List[Image.Image]]:
    image_obs = obs["image"]
    mission_text = obs["mission"]
    direction_idx = obs.get("direction")
    direction_sentence = (
        f"You are currently {DIRECTION_STRINGS.get(direction_idx, 'facing an unknown direction')} in the maze."
        if direction_idx is not None
        else ""
    )

    prompt_text = (
        "Mission: "
        f"{mission_text}. {direction_sentence} "
        "You must choose the next discrete action. Only these actions are valid: "
        "0 (turn left), 1 (turn right), 2 (move forward). Returning 3 (pick up object), 4 (drop), "
        "5 (toggle), or 6 (done) will be treated as no-ops in this level. "
        "Reply with a single digit from 0 to 6 indicating your action."
    )

    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the maze from this image:"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    return chat, [Image.fromarray(image_obs)]


# =================================================================================
# 3: Setup and Environment Creation
# =================================================================================
print("--- Setting up BabyAI Environment and LVLM Components ---")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a single & wrapped environment for the manual rollout
env = LLMProcessingWrapper(
    env=gym.make(ENV_ID, render_mode="rgb_array"),
    processor=processor,
    prompt_formatter=babyai_prompt_formatter,
    max_length=TOKENIZER_MAX_LENGTH,
)


# =================================================================================
# 4: PPO Training Setup and Execution
# =================================================================================
print("\n--- Setting up for PPO Training ---")

# Create the vectorized environment for training
wrapper_kwargs = {
    "processor": processor,
    "prompt_formatter": babyai_prompt_formatter,
    "max_length": TOKENIZER_MAX_LENGTH,
}
vec_env = make_vec_env(
    ENV_ID,
    n_envs=2,
    wrapper_class=lambda e: LLMProcessingWrapper(e, **wrapper_kwargs),
)

# PPO agent configuration
N_STEPS = 512
BATCH_SIZE = 4
N_EPOCHS = 1
LEARNING_RATE = 2e-6
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = PPO(
    device=device,
    policy=LVLMActorCriticPolicy,
    env=vec_env,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    verbose=1,
    tensorboard_log="./ppo_lvlm_babyai/",
    policy_kwargs={
        "device": device,
        "model_name": MODEL_NAME,
        "use_lora": False,
        "exploration_temperature": 7.0,
    },
    clip_range_vf=0.2,
    clip_range=0.2,
    max_grad_norm=1,
    normalize_advantage=False,
    ent_coef=0.001,
    gamma=0.94,
)

# =================================================================================
# 5: PPO Training with Visualization Callback
# =================================================================================

# The callback will save a frame from the first parallel environment every 20 steps
render_callback = SimpleRenderCallback(render_freq=1, image_path=LOGDIR)
vec_env.render_mode = "rgb_array"

try:
    print("\n==================================================")
    print("             STARTING PPO TRAINING              ")
    print("==================================================")
    model.learn(total_timesteps=200000, callback=render_callback)
    print("\n--- Training Finished ---")
    model.save("ppo_lvlm_babyai")
    print("\nModel saved to ppo_lvlm_babyai.zip")
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
finally:
    vec_env.close()
    print("Training environment closed.")
