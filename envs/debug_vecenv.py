import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import Process, Pipe
import time

# ==============================================================================
# 1. A Minimal Mock Environment that Mirrors Your `HumanRGBRacingEnv`
# ==============================================================================
class MockContainedEnv(gym.Env):
    """
    This class perfectly mimics the structural problem of HumanRGBRacingEnv.
    It is a custom gym.Env that, internally, creates and holds another
    time-limited environment instance.
    """
    def __init__(self):
        super().__init__()
        # The core of the problem: a TimeLimit wrapper is created INTERNALLY.
        self.internal_env = gym.make("CartPole-v1", max_episode_steps=100)
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space

    def step(self, action):
        return self.internal_env.step(action)

    def reset(self, seed=None, options=None):
        return self.internal_env.reset(seed=seed, options=options)

    def close(self):
        self.internal_env.close()

# ==============================================================================
# 2. A Dummy Outer Wrapper to Replicate Your Wrapper Chain
# ==============================================================================
class DummyOuterWrapper(gym.Wrapper):
    """Represents the FrameSkip -> LLMProcessingWrapper chain."""
    def __init__(self, env):
        super().__init__(env)

# ==============================================================================
# 3. The Forensic Check Function (Runs in the Subprocess)
# ==============================================================================
def check_env_in_subprocess(pipe, env_fn):
    """
    This function runs in the child process. It creates the environment
    and then inspects its internal structure to verify the timeout.
    """
    try:
        # This is what SubprocVecEnv does: it calls the function it received.
        env = env_fn()
        
        # --- THE CRITICAL INSPECTION ---
        # We need to dig through the wrappers to find the timeout.
        # The structure is: DummyOuterWrapper -> MockContainedEnv -> internal_env
        
        # 1. Get the MockContainedEnv instance
        contained_env = env.env 
        
        # 2. Get its internal_env, which should be the TimeLimit wrapper
        internal_env_with_limit = contained_env.internal_env
        
        # 3. Check if it's actually a TimeLimit wrapper and get its max steps
        if isinstance(internal_env_with_limit, TimeLimit):
            max_steps = internal_env_with_limit._max_episode_steps
            print(f"    [CHILD] SUCCESS: Found TimeLimit wrapper. max_steps = {max_steps}")
            assert max_steps == 100, f"max_steps is {max_steps}, not 100!"
        else:
            # If we get here, the TimeLimit wrapper was stripped away.
            print(f"    [CHILD] FAILURE: internal_env is a {type(internal_env_with_limit)}, NOT TimeLimit!")
            assert False, "The TimeLimit wrapper was lost during serialization!"

        env.close()
        pipe.send(True) # Signal success
    except Exception as e:
        pipe.send(e) # Signal failure by sending the exception
    finally:
        pipe.close()

# ==============================================================================
# 4. Main Test Execution
# ==============================================================================
if __name__ == '__main__':
    
    # --- TEST 1: The Broken (Two-Part) Method ---
    print("==========================================================")
    print("  TESTING THE BROKEN (TWO-PART) `make_vec_env` METHOD   ")
    print("==========================================================")
    
    parent_conn_broken, child_conn_broken = Pipe()
    # This is the function `SubprocVecEnv` will build internally
    broken_fn = make_vec_env(
        MockContainedEnv, # The base env ID/class
        n_envs=1,
        wrapper_class=DummyOuterWrapper, # The wrapper to apply AFTER creation
        vec_env_cls=SubprocVecEnv
    ).env_fns[0]
    
    p_broken = Process(target=check_env_in_subprocess, args=(child_conn_broken, broken_fn))
    p_broken.start()
    result_broken = parent_conn_broken.recv()
    p_broken.join()
    
    if isinstance(result_broken, bool) and result_broken:
        print("\n[UNEXPECTED] The broken method somehow passed.")
    else:
        print(f"\n[EXPECTED] The broken method failed with error: \n    {result_broken}")
        print("\nThis PROVES that splitting the creation process causes the internal TimeLimit to be lost.")

    # --- TEST 2: The Robust (Single-Part) Method ---
    print("\n==========================================================")
    print("   TESTING THE ROBUST (SINGLE-PART) `make_vec_env` METHOD  ")
    print("==========================================================")
    
    parent_conn_fixed, child_conn_fixed = Pipe()
    # Here, we create one single function that does everything.
    robust_fn = make_vec_env(
        lambda: DummyOuterWrapper(MockContainedEnv()), # The factory function
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    ).env_fns[0]

    p_fixed = Process(target=check_env_in_subprocess, args=(child_conn_fixed, robust_fn))
    p_fixed.start()
    result_fixed = parent_conn_fixed.recv()
    p_fixed.join()

    if isinstance(result_fixed, bool) and result_fixed:
        print("\n[SUCCESS] The robust method passed the check. The TimeLimit wrapper was preserved.")
    else:
        print(f"\n[UNEXPECTED] The robust method failed with error: {result_fixed}")