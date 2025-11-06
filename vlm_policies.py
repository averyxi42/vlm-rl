# lvlm_policy.py
from typing import Callable, Dict, Any,Tuple,List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from PIL import Image

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from transformers import AutoProcessor, AutoModelForImageTextToText #BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# =================================================================================
# Component 1: The Pre-processing Wrapper
# =================================================================================
PromptFormatter = Callable[[Any], Tuple[List[Dict], List[Image.Image]]] #type hint

class LLMProcessingWrapper(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that dynamically creates its observation space based
    on the output of a Hugging Face processor. This makes it robust to
    different model architectures and processor implementations.
    """
    def __init__(self, env: gym.Env, processor: AutoProcessor, prompt_formatter:PromptFormatter,max_length:int):
        super().__init__(env)
        self.processor = processor
        self.processor.tokenizer.padding_side = 'left' #enforce left padding

        self.prompt_formatter = prompt_formatter
        self.max_length = max_length
        
        # --- Define the NEW observation space DYNAMICALLY ---
        self.observation_space = self._create_dynamic_observation_space()
    def _get_batched_sample(self, batch_size: int = 2) -> dict:
        """
        Samples a space multiple times and stacks the results into a single
        batched dictionary or array, mimicking a VecEnv's output.
        """
        # This is the core logic in a single, powerful list comprehension
        samples = [self.env.observation_space.sample() for _ in range(batch_size)]
        
        # Check if the space is a dictionary
        if isinstance(self.env.observation_space, spaces.Dict):
            # Stack the values for each key across all sampled dictionaries
            return {key: np.stack([s[key] for s in samples]) for key in samples[0].keys()}
        else:
            # For simple spaces like Box, just stack the arrays
            return np.stack(samples)
    def _create_dynamic_observation_space(self) -> spaces.Dict:
        """
        Performs a "dry run" of the processor to determine the exact structure,
        shapes, and dtypes of its output, then creates a matching Gym space.
        """
        print("[LLMProcessingWrapper] Performing dry run to build observation space...")
        
        # Create a dummy observation from the base environment's space
        dummy_raw_obs = self.env.observation_space.sample()
        # dummy_obs_batch = [self.env.observation_space.sample(), self.env.observation_space.sample()]

        # Run the processing logic once to get a sample output
        # We pass a dummy image and text to the processor
        sample_processed_output = self._process_observation(dummy_raw_obs)
        
        # Now, build the Gym space dictionary from the sample output
        space_dict = {}
        for key, value in sample_processed_output.items():
            space_dict[key] = spaces.Box(
                low=np.iinfo(value.dtype).min if np.issubdtype(value.dtype, np.integer) else -np.inf,
                high=np.iinfo(value.dtype).max if np.issubdtype(value.dtype, np.integer) else np.inf,
                shape=value.shape,
                dtype=value.dtype
            )
            print(f"  - Detected space '{key}': shape={value.shape}, dtype={value.dtype}")
            
        return spaces.Dict(space_dict)

    # def _process_observation(self, obs: Any) -> dict[str, np.ndarray]:
    #     """
    #     A helper method containing the core processing logic.
    #     This is used by both the dry run and the actual observation method.
    #     """
    #     chat, images = self.prompt_formatter(obs)
        
    #     # 2. Apply the model-specific chat template
    #     prompt_with_template = self.processor.tokenizer.apply_chat_template(
    #         chat, tokenize=False, add_generation_prompt=True
    #     )
        
    #     # 3. Process the image and templated prompt
    #     # The max_length can be determined from the tokenizer
        
    #     inputs = self.processor(
    #         text=prompt_with_template,
    #         images=images,
    #         return_tensors="np",
    #         padding="max_length",
    #         max_length=self.max_length,
    #         truncation=True
    #     )
        
    #     # The processor's output is already a dictionary. We just need to
    #     # ensure the dtypes are consistent for the observation space.
    #     # Let's cast integer types to a standard int64 for robustness.
    #     processed_obs = {}
    #     for key, value in inputs.items():
    #         # print(f"{key}:{type(value)}")
    #         if np.issubdtype(value.dtype, np.integer):
    #             processed_obs[key] = value.astype(np.int64)
    #         elif np.issubdtype(value.dtype, np.floating):
    #             processed_obs[key] = value.astype(np.float32)
    #         else:
    #             processed_obs[key] = value

    #     return processed_obs
    def _process_observation(self, obs: Any) -> dict[str, np.ndarray]:
        """
        A helper method containing the core processing logic.
        This is used by both the dry run and the actual observation method.
        """
        chat, images = self.prompt_formatter(obs)

        # 2. Apply the model-specific chat template
        prompt_with_template = self.processor.tokenizer.apply_chat_template(
            chat, tokenize=False, continue_final_message=True#add_generation_prompt=True
        )

        # 3. Process the image and templated prompt
        # The max_length can be determined from the tokenizer
        inputs = self.processor(
            text=prompt_with_template,
            images=images,
            return_tensors="pt",  # Request PyTorch tensors
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # The processor's output is now a dictionary of PyTorch tensors.
        # We need to convert them to NumPy arrays and ensure consistent dtypes.
        processed_obs = {}
        for key, value in inputs.items():
            # For robustness, handle tensors that might be on a GPU or have gradients
            if isinstance(value, torch.Tensor):
                # Detach the tensor from the computation graph and move it to the CPU before converting.
                numpy_value = value.detach().cpu().numpy()
            else:
                # If the value is not a tensor, assume it's already a NumPy array or a compatible type
                numpy_value = np.asanyarray(value)
            try:
                squeezed_value = numpy_value.squeeze(0) #remove batch dim
                numpy_value = squeezed_value
            except:
                # print(f"cannot squeeze {key}")
                pass
            # # Cast integer types to a standard int64 and floats to float32 for robustness.
            # if np.issubdtype(numpy_value.dtype, np.integer):
            #     processed_obs[key] = numpy_value.astype(np.int32)
            # elif np.issubdtype(numpy_value.dtype, np.floating):
            #     processed_obs[key] = numpy_value.astype(np.float32)
            # else:
            #     processed_obs[key] = numpy_value
            processed_obs[key]=numpy_value

        return processed_obs

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """
        Takes a raw observation from the underlying environment and processes it
        using the standardized processing logic.
        """
        return self._process_observation(obs)

# =================================================================================
# Component 2: The LVLM Actor-Critic Policy
# =================================================================================

class LVLMActorCriticPolicy(ActorCriticPolicy):
    """
    A custom Actor-Critic policy for Stable Baselines 3 that uses an LVLM
    as its core feature extractor and policy network.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        model_name: str= "google/gemma-3-4b-it",
        lora_config = None,
        exploration_temperature = 7,
        model_kwargs = {},
        processor_kwargs={},
        detach_value_head = True,
        use_policy_head = False,
        policy_token_idx = -1,#position of the action latents in the sequence, typically last token of prompt
        value_token_idx = -1, #position of the value latents in the sequence, can be a special token after prompt
        **kwargs,
    ):
        # Disable the default MLP extractor; the LVLM is our network.
        kwargs["net_arch"] = []
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # --- 1. Load LVLM  ---
        self.lvlm = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # Recommended for modern models
            **model_kwargs
        )
        self.exploration_temperature = exploration_temperature
        self.detach_value_head = detach_value_head
        self.policy_idx = policy_token_idx
        self.value_idx = value_token_idx
        # --- 2. Add Lora
        if lora_config is not None:
            self.lvlm = get_peft_model(self.lvlm, lora_config)
            print("LoRA enabled. Trainable parameters:")
            self.lvlm.print_trainable_parameters()
        else:
            print("no Lora Config Provided, freezing LVLM")
            for parameter in self.lvlm.parameters():
                parameter.requires_grad = False
        # --- 2. Define the separate Value Head ---
        # The value is estimated from the LVLM's final hidden state.
        # self.value_net = nn.Linear(self.lvlm.config.text_config.hidden_size, 1,dtype=torch.bfloat16)
        hidden_size = self.lvlm.config.text_config.hidden_size
        HEAD_DTYPE = torch.float32
        self.value_net = nn.Sequential(
            # 1. LayerNorm to stabilize the input from the LVLM
            nn.LayerNorm(hidden_size,dtype=HEAD_DTYPE),
            # 2. A deeper MLP to increase capacity
            nn.Linear(hidden_size, 256,dtype=HEAD_DTYPE),
            nn.ReLU(),
            nn.Linear(256, 256,dtype=HEAD_DTYPE),
            nn.ReLU(),
            nn.Linear(256, 1,dtype=HEAD_DTYPE)
        )
        last_layer = self.value_net[-1]
        torch.nn.init.constant_(last_layer.weight, 0)
        torch.nn.init.constant_(last_layer.bias, 0)

        if use_policy_head:
            self.policy_net = nn.Sequential(
            # 1. LayerNorm to stabilize the input from the LVLM
            nn.LayerNorm(hidden_size,dtype=HEAD_DTYPE),
            # 2. A deeper MLP to increase capacity
            nn.Linear(hidden_size, 256,dtype=HEAD_DTYPE),
            nn.ReLU(),
            nn.Linear(256, 256,dtype=HEAD_DTYPE),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n,dtype=HEAD_DTYPE)
            )
            self._get_action_dist = self._get_action_dist_from_head
        else:
            # --- 3. Pre-calculate action token IDs ---
            # This is a critical optimization for action selection.
            tokenizer = AutoProcessor.from_pretrained(model_name,**processor_kwargs).tokenizer
            self.action_token_ids = []
            for i in range(self.action_space.n):
                # We only use the first token if a number is multi-token (e.g., '10').
                # The prompt design should enforce single-digit responses.
                token_id = tokenizer(str(i), add_special_tokens=False).input_ids[0]
                self.action_token_ids.append(token_id)
            
            # Register as a buffer to ensure it's moved to the correct device
            self.register_buffer(
                'action_token_ids_tensor', 
                torch.tensor(self.action_token_ids, dtype=torch.long)
            )

            self._get_action_dist = self._get_action_dist_from_tokens
        # self.to(device)
    # def _build(self, lr_schedule) -> None:
    #     """
    #     Create the networks and the optimizer.

    #     :param lr_schedule: Learning rate schedule
    #         lr_schedule(1) is the initial learning rate
    #     """
    #     param_groups = [
    #         {"params": self.lvlm.parameters(), "lr": 1e-6},  # frozen or tiny lr
    #         {"params": self.value_net.parameters(), "lr": 3e-4},   # **higher LR**
    #         {"params": self.policy_net.parameters(),"lr": 3e-4}
    #     ]
    #     self.optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _prepare_model_input(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepares the observation dictionary for the LVLM.
        This version acts as a robust data-cleaning layer, casting all incoming
        tensors to their correct dtypes as defined in the policy's observation_space.
        This is the user-suggested, optimal solution.
        """
        # Squeeze the extra dimension added by the SB3 data pipeline
        model_input = {k: v for k, v in obs.items()}

        for key, tensor in model_input.items():
            # 1. Get the "source of truth" dtype from this policy's observation space.
            #    This is the NumPy dtype we expect (e.g., np.int32, np.float32).
            correct_numpy_dtype = self.observation_space[key].dtype

            # 2. Determine the required PyTorch dtype for the model.
            if np.issubdtype(correct_numpy_dtype, np.integer):
                # All integer types (input_ids, attention_mask, etc.) must be LongTensors
                # for the model's embedding and indexing layers.
                target_dtype = torch.long
            elif np.issubdtype(correct_numpy_dtype, np.floating):
                # All float types (pixel_values) should match the model's precision.
                target_dtype = torch.bfloat16 # Use the model's native dtype (e.g., bfloat16)
            else:
                # Should not happen with Box spaces, but we can skip if it does.
                continue

            # 3. Cast the tensor ONLY if its current dtype is incorrect.
            if tensor.dtype != target_dtype:
                model_input[key] = tensor.to(target_dtype)

        return model_input

    def _get_value(self,outputs):
        # 3. Get the latent feature for the value function
        latent_vf = outputs.hidden_states[-1][:, self.value_idx, :]
        if self.detach_value_head:
            latent_vf = latent_vf.detach() #prevents gradient backflow
        return self.value_net(latent_vf.float())
    def _get_action_dist_from_tokens(self,outputs):
        """
        Selects the action-specific logits from the LVLM's full vocabulary,
        applies temperature scaling to encourage exploration, and casts to
        float32 for numerical stability before creating the SB3 distribution.
        
        :param outputs: The raw model outputs.
        :return: A Stable Baselines 3 CategoricalDistribution.
        """
        latent_pi = outputs.logits[:, self.policy_idx, :]
        # 1. Select the logits corresponding to our discrete action tokens.
        #    The output will have shape (batch_size, num_actions) and dtype bfloat16.
        action_logits_bf16 = latent_pi[:, self.action_token_ids_tensor]
        # 2. Apply temperature scaling to soften the distribution.
        #    This is the core fix for the low-entropy problem.
        # 3. Explicitly cast to float32.
        #    This is a crucial step for ensuring compatibility and numerical
        #    stability with the SB3 distribution and loss calculation.
        final_action_logits =  action_logits_bf16.float() / self.exploration_temperature
        # 4. Create the final probability distribution.
        #    SB3's distribution class will handle the softmax operation internally.
        return self.action_dist.proba_distribution(action_logits=final_action_logits)

    def _get_action_dist_from_head(self,outputs):
        latent_pi = outputs.hidden_states[-1][:, self.policy_idx, :]
        action_logits = self.policy_net(latent_pi.float())
        final_action_logits =  action_logits / self.exploration_temperature
        return self.action_dist.proba_distribution(action_logits=final_action_logits)

    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass of the policy.
        
        SB3 automatically handles batching and moving the `obs` dictionary's
        tensors to the correct device.
        """
        # The observation from the VecEnv is already a dictionary of batched tensors.
        # We also need to squeeze the extra dimension from the wrapper.
        model_inputs = self._prepare_model_input(obs)

        outputs = self.lvlm(**model_inputs, output_hidden_states=True)

        action_distribution = self._get_action_dist(outputs)
        value = self._get_value(outputs)
        # print(action_distribution.distribution)
        actions = action_distribution.get_actions(deterministic=deterministic)
        log_prob = action_distribution.log_prob(actions)

        return actions, value.float(), log_prob.float()
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Gets the value prediction from the critic part of the network.
        This is a crucial override to prevent the default BasePolicy from
        using the broken FlattenExtractor on our dictionary observation.

        :param obs: The observation dictionary.
        :return: The predicted value of the state.
        """
        model_inputs = self._prepare_model_input(obs)
        with torch.no_grad():
            outputs = self.lvlm(**model_inputs, output_hidden_states=True)
            return self._get_value(outputs)
        
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action from the policy given an observation.
        We override the default implementation to use our custom forward method
        that handles the dict observation and LVLM directly.
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard SB3 method for evaluating actions during training.
        """
        # for k,v in obs.items():
        #     print(f"{k}:{v.shape},{v.dtype}")
        model_input = self._prepare_model_input(obs)
        # for k,v in model_input.items():
        #     print(f"{k}:{v.shape},{v.dtype}")
        outputs = self.lvlm(**model_input, output_hidden_states=True)

        value = self._get_value(outputs)
        action_distribution = self._get_action_dist(outputs)

        log_prob = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        return value.float(), log_prob.float(), entropy.float()
    
class MacroActionWrapper(gym.Wrapper):
    """
    A wrapper that redefines the action space to be a "macro-action" sequence
    of tokens. It decodes this sequence and applies penalties.
    """
    def __init__(
        self, 
        env: gym.Env, 
        tokenizer,
        max_macro_action_len: int,
        no_action_penalty: float,
        multi_action_penalty: float
    ):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.max_macro_action_len = max_macro_action_len
        self.no_action_penalty = no_action_penalty
        self.multi_action_penalty = multi_action_penalty

        # --- 1. Redefine the Action Space ---
        # The action space is now a sequence of token IDs.
        self.action_space = spaces.Box(
            low=0,
            high=tokenizer.vocab_size,
            shape=(self.max_macro_action_len,),
            dtype=np.int64
        )
        
        # --- 2. Map Base Env Actions to Token IDs ---
        # Assumes the base env has a Discrete action space
        assert isinstance(self.env.action_space, spaces.Discrete)
        self.action_token_ids = {
            tokenizer.convert_tokens_to_ids(str(i)) for i in range(self.env.action_space.n)
        }
        self.token_to_env_action_map = {
            tokenizer.convert_tokens_to_ids(str(i)): i for i in range(self.env.action_space.n)
        }

    def step(self, action: np.ndarray):
        """
        Decodes the macro-action, applies penalties, and steps the underlying env.
        
        :param action: A NumPy array of token IDs from the policy.
        """
        # --- 1. Decode the Macro-Action ---
        # Find all valid action tokens in the generated sequence.
        # We ignore padding tokens (usually ID 0 or a specific pad_token_id).
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        action_tokens_in_sequence = [
            token_id for token_id in action if token_id in self.action_token_ids and token_id != pad_token_id
        ]
        decoded_action = self.tokenizer.decode(action,skip_special_tokens=True)
        if decoded_action == "":
            decoded_action = self.tokenizer.decode(action,skip_special_tokens=False)
        print(decoded_action)
        num_actions = len(action_tokens_in_sequence)
        reward_penalty = 0.0
        
        # --- 2. Determine Environment Action and Penalties ---
        if num_actions == 0:
            # No action token found: apply penalty and terminate.
            reward_penalty = self.no_action_penalty
            # Use a default "do nothing" action for the final observation.
            env_action = 0 
 # Terminate, don't truncate
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            # terminated = False
            # truncated = False
            reward += reward_penalty
            print("error! no action token!")
            print(self.action_token_ids)
            print(action)
            return obs, reward, terminated, truncated, info

        elif num_actions > 1:
            # Multiple action tokens found: apply penalty.
            print("error! more than one action")
            reward_penalty = self.multi_action_penalty

        # --- 3. Step the Environment ---
        # Use the LAST valid action token found in the sequence.
        last_action_token = action_tokens_in_sequence[-1]
        env_action = self.token_to_env_action_map[last_action_token]
        print(f"env action: {env_action}")
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        # Apply the penalty to the reward from the environment
        reward += reward_penalty
        
        return obs, reward, terminated, truncated, info
    
# generative_policy.py
from typing import Callable, Dict, Any, Tuple, List, Optional
import torch
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import AutoProcessor, AutoModelForCausalLM
# ... other imports from your previous file ...

class GenerativeLVLMPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box, # Now a Box space for the sequence
        lr_schedule: Callable[[float], float],
        model_name: str = "google/gemma-3-4b-it",
        use_lora: bool = True,
        use_4bit: bool = False,
        exploration_temperature: float = 7.0,
        max_new_tokens: int = 10,
        model_kwargs = {},
        processor_kwargs={},
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.exploration_temperature = exploration_temperature
# Disable the default MLP extractor; the LVLM is our network.
        kwargs["net_arch"] = []
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # --- 1. Load LVLM with Quantization and LoRA for efficiency ---
        # quantization_config = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None

        self.lvlm = AutoModelForImageTextToText.from_pretrained(
            model_name,
            # quantization_config=quantization_config,
            torch_dtype=torch.bfloat16, # Recommended for modern models
            **model_kwargs
          # Handles device placement
        )
        self.exploration_temperature = exploration_temperature
        if use_lora:
            # A standard LoRA configuration
            lora_config = LoraConfig(
                r=128,
                lora_alpha=256,
                lora_dropout=0.05,
                target_modules=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
                task_type="CAUSAL_LM",
            )
            self.lvlm = get_peft_model(self.lvlm, lora_config)
            print("LoRA enabled. Trainable parameters:")
            self.lvlm.print_trainable_parameters()

        # --- 2. Define the separate Value Head ---
        # The value is estimated from the LVLM's final hidden state.
        # self.value_net = nn.Linear(self.lvlm.config.text_config.hidden_size, 1,dtype=torch.bfloat16)
        hidden_size = self.lvlm.config.text_config.hidden_size
        self.value_net = nn.Sequential(
            # 1. LayerNorm to stabilize the input from the LVLM
            nn.LayerNorm(hidden_size,dtype=torch.bfloat16),
            
            # 2. A deeper MLP to increase capacity
            nn.Linear(hidden_size, 256,dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(256, 256,dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(256, 1,dtype=torch.bfloat16)
        )

        last_layer = self.value_net[-1]
        torch.nn.init.constant_(last_layer.weight, 0)
        torch.nn.init.constant_(last_layer.bias, 0)
        # --- 3. Pre-calculate action token IDs ---
        # This is a critical optimization for action selection.
        self.processor = AutoProcessor.from_pretrained(model_name,**processor_kwargs)
        self.processor.tokenizer.padding_side = 'left' #enforce left padding
        # tokenizer = self.processor.tokenizer
        self.action_token_ids = []
        # print(self.action_space)
        # self.to(device)
    def _prepare_model_input(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # (Identical to your previous code)
        model_input = {k: v for k, v in obs.items()}
        if 'input_ids' in model_input:
            model_input['input_ids'] = model_input['input_ids'].long()
        return model_input

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        The ACTOR method. Generates a macro-action and returns all necessary
        data for the RolloutBuffer.
        """
        model_input = self._prepare_model_input(obs)

        # --- 1. Get Value from the initial state (one forward pass) ---
        # This is efficient as it doesn't involve generation.
        with torch.no_grad():
            outputs = self.lvlm(**model_input, output_hidden_states=True)
            latent_vf = outputs.hidden_states[-1][:, -1, :]
            values = self.value_net(latent_vf).float()

        # --- 2. Generate the Macro-Action ---
        # Use the efficient `generate` method which leverages the KV cache.
        gen_outputs = self.lvlm.generate(
            **model_input,
            max_new_tokens=self.max_new_tokens,
            temperature=(1.0 if deterministic else self.exploration_temperature),
            do_sample=not deterministic,
            output_scores=True, # MUST be True to get logits for log_prob calculation
            return_dict_in_generate=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        sequences = gen_outputs.sequences
        scores = gen_outputs.scores # A tuple of logits for each generated step

        # Extract the generated part (the macro-action)
        prompt_len = model_input['input_ids'].shape[1]
        generated_tokens = sequences[:, prompt_len:]

        # --- 3. Calculate the Log Probability of the Generated Sequence ---
        # log_probs = self._calculate_log_prob_of_sequence(scores, generated_tokens)
        generated_mask = (generated_tokens != self.processor.tokenizer.pad_token_id).long()
        log_probs = self._calculate_log_prob_of_sequence(scores, generated_tokens, generated_mask)
        # Pad the generated sequence to the full length of the action space
        actions = F.pad(
            generated_tokens,
            (0, self.max_new_tokens - generated_tokens.shape[1]),
            value=self.processor.tokenizer.pad_token_id
        )

        return actions, values, log_probs

# In your GenerativeLVLMPolicy class

# ... (__init__ and the efficient `forward` method are unchanged) ...

# In your GenerativeLVLMPolicy class

# ... (__init__ and the efficient `forward` method are unchanged) ...

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        The EVALUATOR method for training. 
        This version robustly assembles the input by processing the macro-action
        with RIGHT-PADDING and concatenating the resulting TensorDicts.
        """
        # `obs` is the dict of initial S_env from the buffer.
        # `actions` is the macro-action tensor from the buffer.
        
        # --- 1. PREPARE THE TWO TENSOR DICTIONARIES ---
        
        # a) The observation dictionary (S_env) is already correctly (left-)padded by the wrapper
        obs_dict = {k: v for k, v in obs.items()}
        # print("obs dict")
        # for k,v in obs_dict.items():
        #     print(f"{k}:{v.shape},{type(v)}")
        # b) The macro-action dictionary (A_macro)
        # Convert the macro-action token IDs back to text strings.
        macro_action_texts = self.processor.batch_decode(actions.long().cpu(), skip_special_tokens=True)
        
        # --- THE CRUCIAL STEP: ENSURE RIGHT-PADDING FOR ACTIONS ---
        # Save the original padding side to be a good citizen
        try:
            # Temporarily set the tokenizer to right-pad
            self.processor.tokenizer.padding_side = 'right'
            
            action_dict = self.processor(
                text=macro_action_texts,
                return_tensors="pt",
                padding="max_length", # Use right-padding as now configured
                max_length = self.max_new_tokens,
                add_special_tokens=False,
            ).to(self.device)
        finally:
            # Always restore the original padding side
            self.processor.tokenizer.padding_side = "left"

        # --- 2. CONCATENATE THE TENSORDICTS KEY-WISE ---
        
        prompt_len = obs_dict['input_ids'].shape[1]
        
        final_input = {}
        
        # Find the set of keys that are common to both dictionaries
        common_keys = obs_dict.keys() & action_dict.keys()
        
        # For all common keys, concatenate the tensors along the sequence dimension
        for key in common_keys:
            final_input[key] = torch.cat([obs_dict[key], action_dict[key]], dim=1)
            
        # For keys that are *only* in the observation dictionary (like 'pixel_values'),
        # simply copy them over.
        for key in obs_dict.keys() - common_keys:
            final_input[key] = obs_dict[key]
        
        decoded_ids = self.processor.batch_decode(final_input['input_ids'].cpu(), skip_special_tokens=False)
        # print(f"training with the following batch:")
        # print(decoded_ids)
        # --- 3. THE SINGLE, EFFICIENT FORWARD PASS ---
        # print("final_input")
        # for k,v in final_input.items():
        #     print(f"{k}:{v.shape},{type(v)}")
        outputs = self.lvlm(**final_input, output_hidden_states=True)

        # --- 4. EXTRACT OUTPUTS ---
        
        # Value from the state *before* the macro-action (index: prompt_len - 1)
        latent_vf = outputs.hidden_states[-1][:, prompt_len - 1, :]#.detach()
        values = self.value_net(latent_vf).float()

        # Logits for the positions where the macro-action tokens were generated
        # We slice the logits from the end of the prompt to cover the action sequence length
        action_len = action_dict['input_ids'].shape[1]
        logits_for_actions = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]
        
        # Calculate log_prob and entropy using our helper methods
        # We must pass the correctly padded action_dict['input_ids']
        log_probs = self._calculate_log_prob_of_sequence(logits_for_actions, action_dict['input_ids'],action_dict['attention_mask'])
        entropy = self._calculate_sequence_entropy(logits_for_actions)

        return values, log_probs, entropy
    # def _calculate_log_prob_of_sequence(self, logits: Tuple[torch.Tensor], sequence: torch.Tensor,mask=None) -> torch.Tensor:
    #     """
    #     Helper to calculate the total log probability of a generated sequence.
        
    #     :param logits: A tuple of tensors (from `generate`) or a single tensor (from `evaluate`).
    #     :param sequence: The sequence of token IDs.
    #     """
    #     if isinstance(logits, tuple):
    #         # Logits from `generate` are per-step
    #         log_probs = []
    #         for i, step_logits in enumerate(logits):
    #             step_log_probs = F.log_softmax(step_logits, dim=-1)
    #             token_id = sequence[:, i]
    #             # Gather the log_prob of the specific token that was chosen
    #             log_probs.append(step_log_probs.gather(1, token_id.unsqueeze(-1)))
    #         return torch.cat(log_probs, dim=1).sum(dim=1)
    #     else:
    #         # Logits from `evaluate_actions` are a single tensor
    #         log_probs = F.log_softmax(logits, dim=-1)
            
    #         return log_probs.gather(2, sequence.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    def _calculate_log_prob_of_sequence(
        self, logits: Tuple[torch.Tensor], sequence: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Helper to calculate the MASKED total log probability."""
        if isinstance(logits, tuple):
            log_probs_list = []
            for i, step_logits in enumerate(logits):
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                token_id = sequence[:, i]
                log_probs_list.append(step_log_probs.gather(1, token_id.unsqueeze(-1)))
            log_probs_per_step = torch.cat(log_probs_list, dim=1)
        else:
            log_probs_per_step = F.log_softmax(logits, dim=-1).gather(2, sequence.unsqueeze(-1)).squeeze(-1)

        # --- FIX: Apply the mask before summing ---
        masked_log_probs = log_probs_per_step * mask
        return masked_log_probs.sum(dim=1)
    def _calculate_sequence_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Helper to calculate the entropy of the distributions."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Entropy H(p) = - sum(p * log(p))
        return -(probs * log_probs).sum(dim=-1).mean() # Return mean entropy over the sequence

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Gets the value prediction from the critic part of the network.
        This is a crucial override to prevent the default BasePolicy from
        using the broken FlattenExtractor on our dictionary observation.

        :param obs: The observation dictionary.
        :return: The predicted value of the state.
        """
        # We define the correct forward path to get the value,
        # which is to go through the LVLM and then the value_net.
        obs = self._prepare_model_input(obs)
        with torch.no_grad():
            # 1. Prepare model input, same as in `forward()`
            model_input = {k: v.squeeze(1) for k, v in obs.items()}
            
            # 2. Forward pass through the LVLM to get hidden states
            outputs = self.lvlm(**model_input, output_hidden_states=True)
            
            # 3. Get the latent feature for the value function
            latent_vf = outputs.hidden_states[-1][:, -1, :]
            
            # 4. Pass it through our value head
            return self.value_net(latent_vf).float()
