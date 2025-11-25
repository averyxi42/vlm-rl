import gymnasium
import numpy as np
from typing import Any, Tuple, Optional

class HabitatRenderWrapper(gymnasium.Wrapper):
    """
    A simple wrapper to make the Habitat env's 'rgb' observation
    available via the standard gymnasium render() method.
    
    This fixes the 'AttributeError: can't set attribute 'render_mode''
    by overriding the base wrapper's read-only 'render_mode' property.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        # --- THIS IS THE FIX ---
        # We can't SET self.render_mode.
        # Instead, we just set an internal variable.
        self._render_mode = "rgb_array"
        # ---------------------
        
        # Add "rgb_array" to metadata so SB3/other tools know it's available
        new_modes = self.metadata.get("render_modes", []).copy()
        if "rgb_array" not in new_modes:
            new_modes.append("rgb_array")
        self.metadata["render_modes"] = new_modes
        
        # Cache for the last observation
        self.last_rgb_obs = None

    # --- THIS IS THE FIX ---
    # We override the read-only property from gymnasium.Wrapper
    @property
    def render_mode(self) -> str | None:
        """Returns the render mode of this wrapper."""
        return self._render_mode
    # ---------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Cache the RGB observation
        self.last_rgb_obs = obs.get("rgb")
        
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Cache the RGB observation
        self.last_rgb_obs = obs.get("rgb")
        
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns the last cached RGB observation.
        """
        # This check will now use our custom @property
        if self.render_mode != "rgb_array":
            raise ValueError(
                f"This wrapper only supports render_mode='rgb_array', "
                f"not {self.render_mode}"
            )
        
        if self.last_rgb_obs is None:
            print("Warning: render() called before reset(). Returning blank image.")
            rgb_space = self.observation_space.spaces.get("rgb")
            if rgb_space:
                return np.zeros(rgb_space.shape, dtype=rgb_space.dtype)
            else:
                return np.zeros((256, 256, 3), dtype=np.uint8)

        return self.last_rgb_obs
    
import cv2  # <-- We now need OpenCV

class HabitatRenderWrapperFancy(gymnasium.Wrapper):
    """
    Enhanced render wrapper for Habitat.
    
    - Provides a "rgb_array" render mode.
    - Caches the 'rgb' and 'top_down_map' from the observation.
    - On render(), it resizes the map to match the RGB view's height
      and returns a new image with them stacked side-by-side.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        self._render_mode = "rgb_array"
        
        # Add "rgb_array" to metadata
        new_modes = self.metadata.get("render_modes", []).copy()
        if "rgb_array" not in new_modes:
            new_modes.append("rgb_array")
        self.metadata["render_modes"] = new_modes
        
        # Caches for both observations
        self.last_rgb_obs = None
        self.last_map_obs = None
        self.blank_image = None # Cache for a blank image if needed

    @property
    def render_mode(self) -> str | None:
        """Returns the render mode of this wrapper."""
        return self._render_mode

    def _cache_obs(self, obs: dict):
        """Helper to cache the RGB and Map observations."""
        self.last_rgb_obs = obs.get("rgb")
        self.last_map_obs = obs.get("top_down_map")
        
        # Create a blank image cache if this is the first time
        if self.blank_image is None and self.last_rgb_obs is not None:
            self.blank_image = np.zeros_like(self.last_rgb_obs)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        
        obs, info = self.env.reset(seed=seed, options=options)
        self._cache_obs(obs)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._cache_obs(obs)
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns a side-by-side view of the agent's RGB observation
        and the top-down map.
        """
        if self.render_mode != "rgb_array":
            raise ValueError(
                f"This wrapper only supports render_mode='rgb_array', "
                f"not {self.render_mode}"
            )
        
        # Handle if render() is called before reset()
        if self.last_rgb_obs is None:
            print("Warning: render() called before reset(). Returning blank image.")
            # Try to build a blank from the space
            if self.blank_image is None:
                rgb_space = self.observation_space.spaces.get("rgb")
                if rgb_space:
                    self.blank_image = np.zeros(rgb_space.shape, dtype=rgb_space.dtype)
                else:
                    # Failsafe
                    self.blank_image = np.zeros((256, 256, 3), dtype=np.uint8)
            return self.blank_image

        # If we have no map, just return the RGB view
        if self.last_map_obs is None:
            return self.last_rgb_obs

        # --- Image Combination Logic ---
        rgb_img = self.last_rgb_obs
        map_img = self.last_map_obs

        # Ensure map is 3-channel RGB (it might be RGBA)
        if map_img.shape[2] == 4:
            map_img = cv2.cvtColor(map_img, cv2.COLOR_RGBA2RGB)

        # Get target height from RGB view
        h, w, _ = rgb_img.shape
        h_map, w_map, _ = map_img.shape

        # Calculate new map width to maintain aspect ratio
        new_w_map = int(w_map * (h / h_map))
        
        # Resize map
        resized_map = cv2.resize(map_img, (new_w_map, h), interpolation=cv2.INTER_LINEAR)
        
        # Stack them side-by-side
        combined_img = np.hstack((rgb_img, resized_map))
        
        return combined_img