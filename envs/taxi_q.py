import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm # For a nice progress bar

# --- 1. Set up the Environment ---
env = gym.make("Taxi-v3")

# --- 2. Define Hyperparameters ---
# These values control the learning process.
episodes = 25000       # Total number of games to play for training
learning_rate = 0.2    # (alpha) How much we update our Q-values based on new info
discount_factor = 0.99 # (gamma) How much we value future rewards over immediate ones

# Epsilon-greedy strategy parameters for exploration vs. exploitation
epsilon = 1.0          # Initial probability of taking a random action
epsilon_decay = 0.9999 # How much epsilon decreases after each episode
min_epsilon = 0.01     # The minimum probability of taking a random action

# --- 3. Initialize the Q-table ---
# The Q-table holds the learned values for every state-action pair.
# Rows are states, columns are actions.
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

print(f"Q-table initialized with shape: {q_table.shape}")
print(f"({state_space_size} states x {action_space_size} actions)")

# --- 4. The Q-Learning Training Loop ---
print("\n--- Starting Training ---")
for episode in tqdm(range(episodes)):
    state, info = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            # Explore: take a random action
            action = env.action_space.sample()
        else:
            # Exploit: take the best known action for the current state
            action = np.argmax(q_table[state, :])
            
        # Take the action and observe the outcome
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if(reward>0):
            print("success!")
        # --- The Core Q-Learning Update Rule ---
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :]) # Best Q-value for the next state
        
        # Bellman equation: Update the Q-value for the state-action pair
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value
        
        # Move to the next state
        state = next_state
        
    # Decay epsilon after each episode to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("--- Training Finished ---")

# --- 5. Save the Results ---
# The q_table now contains the learned policy. We can save it for later use.
np.save("taxi_q_table.npy", q_table)
print("\nQ-table saved successfully to 'taxi_q_table.npy'")

env.close()


# =============================================================================
# --- 6. Evaluate the Trained Agent ---
# =============================================================================
print("\n--- Evaluating the trained agent... ---")

# Load the environment in render mode and the saved Q-table
eval_env = gym.make("Taxi-v3", render_mode="rgb_array")
trained_q_table = np.load("taxi_q_table.npy")

for i in range(5): # Run 5 evaluation episodes
    state, info = eval_env.reset()
    done = False
    total_reward = 0
    print(f"\n--- Episode {i+1} ---")
    
    while not done:
        # For evaluation, we ONLY exploit (no random actions)
        action = np.argmax(trained_q_table[state, :])
        state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode finished with total reward: {total_reward}")

eval_env.close()