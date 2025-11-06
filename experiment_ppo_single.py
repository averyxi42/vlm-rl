import argparse
import importlib
import os
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    BaseCallback
)
import imageio
import numpy as np
from vlm_policies import LVLMActorCriticPolicy
from visualization.visualization_utils import SimpleRenderCallback

# ---------------------- Custom Diagnostics Callback ---------------------- #
class LVLMStatsCallback:
    def __init__(self, logdir):
        self.logdir = logdir
        self.step = 0

    def __call__(self, locals_, globals_):
        model = locals_["model"]
        writer = model.logger.tensorboard_writer

        # Value head statistics
        with torch.no_grad():
            params = list(model.policy.value_net.parameters())
            writer.add_scalar("value_head/weight_norm", params[0].norm().item(), self.step)

        self.step += 1
        return True
    
class RolloutGIFCallback(BaseCallback):
    """
    Periodically runs one evaluation episode and saves frames as GIF.
    Works on headless servers.
    """
    def __init__(self, eval_env, save_dir, freq=50_000):
        super().__init__()
        self.eval_env = eval_env
        self.freq = freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.freq != 0:
            return True
        
        frames = []
        obs, _ = self.eval_env.reset()
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.eval_env.step(action)
            frame = self.eval_env.render()  # must return RGB array
            frames.append(np.array(frame))

            if len(frames) > 400:  # safety limit
                break

        outfile = os.path.join(self.save_dir, f"rollout_{self.num_timesteps}.gif")
        imageio.mimsave(outfile, frames, fps=20)

        # Also overwrite "latest.gif" for live viewing
        imageio.mimsave(os.path.join(self.save_dir, "latest.gif"), frames, fps=20)

        print(f"[RolloutGIF] Saved {outfile}")
        return True

# ----------------------------- Main Script ------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config Python module (no .py extension)")
    parser.add_argument("--project", required=True, help="Project name (used for log directory)")
    parser.add_argument("--device",required=False,default='cuda')
    args = parser.parse_args()

    # Import config
    cfg = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))

    # Setup project directory
    BASE = "./runs"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    proj_dir = os.path.join(BASE, args.project, run_id)
    os.makedirs(proj_dir, exist_ok=True)

    print(f"▶ Saving logs & checkpoints to: {proj_dir}")

    # Build vectorized environment
    def make_single_env():
        env = cfg.make_env()
        env = cfg.ENV_WRAPPER(env)
        return env

    from stable_baselines3.common.env_util import make_vec_env
    env = make_vec_env(make_single_env, cfg.n_envs, vec_env_cls=cfg.vec_env_cls)
    env.render_mode='rgb_array'

    # Instantiate PPO with our policy and NO LR schedule interference
    model = PPO(
        LVLMActorCriticPolicy,
        env,
        tensorboard_log=proj_dir,
        **cfg.ppo,
        policy_kwargs=cfg.policy,
        device=args.device,
    )
    cfg.modify_ppo(model)
        
    # Attach custom callbacks
    checkpoint = CheckpointCallback(save_freq=cfg.save_freq, save_path=proj_dir, name_prefix="checkpoint")
    # stats = LVLMStatsCallback(proj_dir)
    liveview = SimpleRenderCallback(1,image_path=proj_dir,image_name="live_view.png")

    # callback = EveryNTimesteps(1000, stats)

    print("🚀 Training started")
    model.learn(total_timesteps=cfg.total_timesteps, callback=[checkpoint, liveview,*cfg.callbacks],progress_bar=True)
    model.save(os.path.join(proj_dir, "final_model"))
    print("✅ Training complete")


if __name__ == "__main__":
    main()
