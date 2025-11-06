
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2 # Using OpenCV for resizing
from typing import Tuple,List,Dict
from PIL import Image
class HumanRGBTaxiEnv(gym.Env):
    """
    A version of the Taxi-v3 environment where the observation is the
    human-readable 'rgb_array' render, resized for deep learning.
    This presents a more realistic perception challenge for the agent.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode='rgb_array',img_shape = [550,350, 3]):
        # We need the internal environment to be in 'rgb_array' mode
        self.internal_env = gym.make("Taxi-v3", render_mode='rgb_array')
        self.render_mode = render_mode
        self.img_shape = img_shape
        # The observation space is a standard 84x84 RGB image
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        self.action_space = self.internal_env.action_space

        self.window = None
        self.clock = None

    def _get_obs(self):
        # Get the RGB array from the underlying environment
        frame = self.internal_env.render()
        
        # Resize to a standard dimension for CNNs
        resized_frame = cv2.resize(frame, self.img_shape[:2], interpolation=cv2.INTER_AREA)
        return resized_frame

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # The internal env's reset doesn't return an observation, so we call render
        self.internal_env.reset(seed=seed, options=options)
        return self._get_obs(), {}

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.internal_env.step(action)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # This render method is for human viewing during inference/evaluation
        if self.render_mode == "human":
            # Get the original, high-res frame for display
            frame = self.internal_env.render()
            if self.window is None:
                import pygame
                pygame.init()
                self.window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                pygame.display.set_caption("HumanRGBTaxiEnv")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else: # "rgb_array"
            return self._get_obs()

    def close(self):
        self.internal_env.close()
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
class OracleHumanRGBTaxiEnv(HumanRGBTaxiEnv):
    """
    An augmented version of the HumanRGBTaxiEnv for debugging purposes.

    This environment is identical to its parent, but it also:
    1. Loads a pre-trained Q-table ('taxi_q_table.npy') upon initialization.
    2. Modifies the observation space to be a dictionary containing both the
       image ("image") and the optimal action according to the Q-table
       ("oracle_action").
    3. At each step, it determines the best action for the current discrete
       state and includes it in the returned observation.
    """
    def __init__(self, render_mode='rgb_array', img_shape=[550, 350, 3]):
        # Initialize the parent class first to set up the internal env
        super().__init__(render_mode=render_mode, img_shape=img_shape)

        # Load the pre-trained Q-table
        try:
            self.q_table = np.load("/Projects/vlm-rl/envs/taxi_q_table.npy")
            print("Oracle Q-table 'taxi_q_table.npy' loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find 'taxi_q_table.npy' in the same directory. "
                "Please run the Q-learning script first to generate it."
            )

        # Overwrite the observation space to be a dictionary
        self.image_space = self.observation_space # Keep a reference to the original image space
        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "oracle_action": spaces.Discrete(self.action_space.n)
        })

    def _get_oracle_action(self):
        """
        Gets the optimal action for the current internal discrete state
        using the loaded Q-table.
        """
        # The true discrete state is stored in the unwrapped internal environment
        current_discrete_state = self.internal_env.unwrapped.s
        # The best action is the one with the highest Q-value for this state
        return np.argmax(self.q_table[current_discrete_state])

    def reset(self, seed=None, options=None):
        # The parent's reset method returns the image observation and info dict
        image_obs, info = super().reset(seed=seed, options=options)
        
        # Get the optimal action for the initial state
        oracle_action = self._get_oracle_action()
        
        # Combine into the new dictionary observation
        dict_obs = {
            "image": image_obs,
            "oracle_action": oracle_action
        }
        
        return dict_obs, info

    def step(self, action):
        # The parent's step method returns the next image_obs, reward, etc.
        image_obs, reward, terminated, truncated, info = super().step(action)
        
        # Get the optimal action for the *new* state we've just entered
        oracle_action = self._get_oracle_action()
        
        # Combine into the new dictionary observation
        dict_obs = {
            "image": image_obs,
            "oracle_action": oracle_action
        }

        return dict_obs, reward, terminated, truncated, info

def taxi_formatter(obs: np.ndarray) -> Tuple[List[Dict], List[Image.Image]]:
    """
    A prompt formatter for the HumanRGBTaxiEnv.
    
    This function converts the visual observation of the Taxi environment into a
    multimodal conversation prompt, designed to be used with a Large 
    Vision-Language Model (LVLM) as the policy.

    The prompt provides the LVLM with:
    1.  The role it should play (a self-driving taxi AI).
    2.  The overall objective of the game.
    3.  A description of the visual elements in the image.
    4.  A clear list of the 6 discrete actions it can choose from.
    5.  Strategic hints to guide its decision-making process.
    6.  Strict formatting requirements for its response.

    Args:
        obs (np.ndarray): The RGB image observation from the environment.

    Returns:
        A tuple containing:
        - A list of dictionaries representing the chat prompt.
        - A list of PIL Images to be included in the prompt.
    """
    
    # prompt_text = (
    #     "You are a self-driving taxi AI. Your goal is to navigate the building to pick up the "
    #     "passenger and drop them off at their destination.\n\n"
    #     "The image is a top-down view of the environment. The yellow car is your taxi. "
    #     "The colored tile next to the building is the drop off location.\n\n"
    #     "Choose ONE of the following actions:\n"
    #     "[action id 0]: Move South (down)\n"
    #     "[action id 1]: Move North (up)\n"
    #     "[action id 2]: Move East (right)\n"
    #     "[action id 3]: Move West (left)\n"
    #     "[action id 4]: Pick up the passenger\n"
    #     "[action id 5]: Drop off the passenger\n\n"
    #     "Follow these rules to decide your action:\n"
    #     "1.  If the passenger is visible AND your taxi is on the same square, you MUST choose action 4 to pick them up.\n"
    #     "2.  If the passenger is in your taxi (the person is NOT visible) AND you are at the correct colored destination, you MUST choose action 5 to drop them off.\n"
    #     "3.  Otherwise, navigate towards the passenger (if not picked up) or the destination (if picked up). The green walls are obstacles you can't move through.\n\n"
    #     # "You MUST respond with a single number corresponding to the action ID, with NO extra formatting or text."
    #     "Briefly reason about the current scene and then pick a single numerical action id."
    # )
    prompt_text = (
        "You are an AI agent solving the classical 'taxi-v3' grid world. Your goal is to navigate to pick up the "
        "passenger and drop them off at their destination.\n\n"
        "The image is a top-down view of the environment. The yellow car is your taxi. "
        "The colored tile next to the building is the drop off location.\n\n"
        "Choose ONE of the following actions:\n"
        "[id 0]: Move down\n"
        "[id 1]: Move up\n"
        "[id 2]: Move right\n"
        "[id 3]: Move left\n"
        "[id 4]: Embark passenger\n"
        "[id 5]: Drop off passenger\n\n"
        "Note that the directions are absolute, NOT relative to your car."
        "Actions 4 is ONLY valid if your car is on the same tile as the passenger i.e they overlap."
        "You MUST NOT choose 4 unless the car and passenger overlap."
        # "You MUST respond with a single number corresponding to the action ID, with NO extra formatting or text."
        "Briefly reason about the current scene and then pick a single NUMERICAL action id. Your must utter NO other digits than the one you intend to choose."
    )

    # prompt_text = "describe what you see."
    
    # The standard chat format for a multimodal prompt
    chat_prompt = [
        # {"role":"system",
        #     "content":[{"type":"text","text":"You are an adept embodied agent in a reinforcement learning setting."}] 
        # },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the state of the taxi environment from this image and choose the next action."},
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": prompt_text},
            ]
        },
        {"role": "assistant", "content": 'After thoroughly analyzing the image, I have determined the optimal action is '},
    ]
    
    # The LVLM processor typically expects a list of PIL Images
    images = [Image.fromarray(obs)]
    
    return chat_prompt, images

def cheater_taxi_formatter(obs):
    """
    A prompt formatter for the HumanRGBTaxiEnv.
    
    This function converts the visual observation of the Taxi environment into a
    multimodal conversation prompt, designed to be used with a Large 
    Vision-Language Model (LVLM) as the policy.

    The prompt provides the LVLM with:
    1.  The role it should play (a self-driving taxi AI).
    2.  The overall objective of the game.
    3.  A description of the visual elements in the image.
    4.  A clear list of the 6 discrete actions it can choose from.
    5.  Strategic hints to guide its decision-making process.
    6.  Strict formatting requirements for its response.

    Args:
        obs (np.ndarray): The RGB image observation from the environment.

    Returns:
        A tuple containing:
        - A list of dictionaries representing the chat prompt.
        - A list of PIL Images to be included in the prompt.
    """
    
    prompt_text = (
        "You are a self-driving taxi AI. Your goal is to navigate the building to pick up the "
        "passenger and drop them off at their destination.\n\n"
        "The image is a top-down view of the environment. The yellow car is your taxi. "
        "The colored tile next to the building is the drop off location.\n\n"
        "Choose ONE of the following actions:\n"
        "[action id 0]: Move South (down)\n"
        "[action id 1]: Move North (up)\n"
        "[action id 2]: Move East (right)\n"
        "[action id 3]: Move West (left)\n"
        "[action id 4]: Pick up the passenger\n"
        "[action id 5]: Drop off the passenger\n\n"
        f"hint: the best action might be {obs['oracle_action']}!"
        "You MUST respond with a single number corresponding to the action ID, with NO extra formatting or text."
    )
    
    # The standard chat format for a multimodal prompt
    chat_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the state of the taxi environment from this image and choose the next action."},
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": prompt_text},
            ]
        }
    ]
    
    # The LVLM processor typically expects a list of PIL Images
    images = [Image.fromarray(obs['image'])]
    
    return chat_prompt, images


