# env_factory.py
import gymnasium as gym
from functools import partial
from transformers import AutoProcessor

from stable_baselines3.common.env_util import make_vec_env
from lvlm_policy import LLMProcessingWrapper, DummyImageEnv # Import your classes

def create_lvlm_env(env_id: str, model_name: str, n_envs: int, prompt_text: str):
    """
    A factory function to create a wrapped and vectorized environment for an LVLM policy.
    
    This function handles the "dry run" to build the observation space dynamically,
    encapsulating the setup complexity.

    :param env_id: The ID of the base gym environment (or a custom class).
    :param model_name: The Hugging Face name of the LVLM.
    :param n_envs: The number of parallel environments to create.
    :param prompt_text: The instruction prompt for the LVLM.
    :return: A ready-to-use Stable Baselines 3 VecEnv.
    """
    print("--- Creating LVLM Environment ---")
    
    # 1. Load the processor
    processor = AutoProcessor.from_pretrained(model_name)

    # 2. Define wrapper keyword arguments
    # We use a partial function to pass these to the wrapper in make_vec_env
    wrapper_kwargs = {
        'processor': processor,
        'prompt_text': prompt_text,
    }
    
    # This is where the magic happens. make_vec_env will instantiate the
    # LLMProcessingWrapper for each environment, and the wrapper's __init__
    # (with our dynamic dry run) will be called automatically.
    if env_id == "DummyImageEnv":
        base_env = DummyImageEnv
    else:
        base_env = env_id

    vec_env = make_vec_env(
        base_env,
        n_envs=n_envs,
        wrapper_class=partial(LLMProcessingWrapper, **wrapper_kwargs)
    )
    
    print("--- Environment Created Successfully ---\n")
    return vec_env