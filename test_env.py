import gym as old_gym  # The library Habitat-gym uses
import gymnasium       # The library SB3/Shimmy uses
import numpy as np
from habitat.config.default import get_config
from habitat import make_dataset
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)
from habitat.gym import make_gym_from_config
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import cv2  # <-- Import OpenCV for saving the image
from typing import Any, Tuple, Optional # For the wrapper

# -----------------------------------------------------------------
# PASTE THE HabitatRenderWrapper CLASS DEFINITION HERE
# -----------------------------------------------------------------
class HabitatRenderWrapper(gymnasium.Wrapper):
    """
    Enhanced render wrapper for Habitat.
    ... (paste the full class code from above) ...
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
        
        if self.last_rgb_obs is None:
            print("Warning: render() called before reset(). Returning blank image.")
            if self.blank_image is None:
                rgb_space = self.observation_space.spaces.get("rgb")
                if rgb_space:
                    self.blank_image = np.zeros(rgb_space.shape, dtype=rgb_space.dtype)
                else:
                    self.blank_image = np.zeros((256, 256, 3), dtype=np.uint8)
            return self.blank_image

        if self.last_map_obs is None:
            return self.last_rgb_obs

        rgb_img = self.last_rgb_obs
        map_img = self.last_map_obs

        if map_img.shape[2] == 4:
            map_img = cv2.cvtColor(map_img, cv2.COLOR_RGBA2RGB)

        h, w, _ = rgb_img.shape
        h_map, w_map, _ = map_img.shape
        new_w_map = int(w_map * (h / h_map))
        resized_map = cv2.resize(map_img, (new_w_map, h), interpolation=cv2.INTER_LINEAR)
        
        combined_img = np.hstack((rgb_img, resized_map))
        
        return combined_img
# -----------------------------------------------------------------
# END of wrapper definition
# -----------------------------------------------------------------


def make_env_for_testing():
    """
    Creates the fully wrapped, patched, and renderable env.
    """
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
    habitat_env = make_gym_from_config(config_env, dataset)

    # 2. Apply the patch
    for key, space in habitat_env.observation_space.spaces.items():
        if isinstance(space, old_gym.spaces.Box):
            if (space.low > space.high).any():
                print(f"   WARNING: Fixing invalid observation space for '{key}'.")
                new_high = np.maximum(space.low, space.high)
                new_space = old_gym.spaces.Box(
                    low=space.low,
                    high=new_high,
                    shape=space.shape,
                    dtype=space.dtype,
                )
                habitat_env.observation_space.spaces[key] = new_space

    # 3. Apply the shimmy wrapper
    shimmy_env = GymV21CompatibilityV0(env=habitat_env)
    
    # 4. Apply the render wrapper
    final_env = HabitatRenderWrapper(shimmy_env)

    return final_env

# --- Main Execution Logic ---

print("1. Creating fully wrapped and patched env...")
env = make_env_for_testing()

print("\n2. Verifying env metadata...")
print(f"   Render Modes: {env.metadata.get('render_modes')}")
print(f"   Render Mode: {env.render_mode}")
assert "rgb_array" in env.metadata["render_modes"]
assert env.render_mode == "rgb_array"
print("   ✅ Render mode checks passed.")

# --- Test reset() ---
print("\n3. Testing reset()...")
obs, info = env.reset()
print("   ✅ reset() successful.")

# --- Test render() after reset ---
print("\n4. Testing render() after reset...")
render_img = env.render()
assert isinstance(render_img, np.ndarray), "render() did not return an array!"
assert len(render_img.shape) == 3 and render_img.shape[2] == 3
print("   ✅ render() returned a valid image array.")

# --- Test step() ---
print("\n5. Testing step()...")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("   ✅ step() successful.")

# --- Test render() after step and save ---
print("\n6. Testing render() after step and saving to file...")
render_img_step = env.render()
assert isinstance(render_img_step, np.ndarray)

# Define file path
output_file = "habitat_render_test.png"

# Convert from RGB (which Habitat/render gives) to BGR (which cv2.imwrite needs)
try:
    bgr_image = cv2.cvtColor(render_img_step, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, bgr_image)
    print(f"   ✅ Successfully saved visualization to '{output_file}'")
    print("   Check this file to verify the side-by-side RGB + Map view.")
except Exception as e:
    print(f"\n   ❌ FAILED to save image: {e}")


print("\n--- VERIFICATION COMPLETE ---")
print("You are now in an interactive IPython shell.")