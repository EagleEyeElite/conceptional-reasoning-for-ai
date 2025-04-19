import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
        print("Starting training with a new Q-table...")
    else:
        try:
            with open('mountain_car.pkl', 'rb') as f:
                q = pickle.load(f)
            print("Loaded existing Q-table from file")
        except FileNotFoundError:
            print("No existing Q-table found. Creating a new one...")
            q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
            is_training = True  # Force training mode if no Q-table exists

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    # For progress tracking
    start_time = time.time()
    progress_interval = max(1, episodes // 20)  # Show progress ~20 times during training
    best_reward = -float('inf')

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        rewards = 0
        steps = 0

        while not terminated and rewards > -1000:
            steps += 1

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                        reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        # Track episode statistics
        rewards_per_episode[i] = rewards
        steps_per_episode[i] = steps

        # Update epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Update best reward
        if rewards > best_reward:
            best_reward = rewards

        # Print progress
        if (i + 1) % progress_interval == 0 or i == 0 or i == episodes - 1:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_per_episode[max(0, i-100):(i+1)])
            success = "Yes" if terminated else "No"

            print(f"Episode {i+1}/{episodes} ({(i+1)/episodes*100:.1f}%) | " +
                  f"ε: {epsilon:.3f} | " +
                  f"Steps: {steps} | " +
                  f"Reward: {rewards:.1f} | " +
                  f"Avg Reward (100): {avg_reward:.1f} | " +
                  f"Best: {best_reward:.1f} | " +
                  f"Success: {success} | " +
                  f"Time: {elapsed_time:.1f}s")

    env.close()

    # Calculate final statistics
    final_avg_reward = np.mean(rewards_per_episode[-100:])
    avg_steps = np.mean(steps_per_episode)
    success_rate = np.sum(rewards_per_episode > -200) / episodes * 100  # Rough estimate of success

    print("\n" + "="*60)
    print(f"Training completed in {time.time() - start_time:.1f} seconds")
    print(f"Final average reward (last 100 episodes): {final_avg_reward:.1f}")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"Estimated success rate: {success_rate:.1f}%")
    print(f"Exploration rate (epsilon): {epsilon:.3f}")
    print("="*60 + "\n")

    # Save Q table to file
    if is_training:
        try:
            with open('mountain_car.pkl', 'wb') as f:
                pickle.dump(q, f)
            print("Q-table saved successfully")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

        try:
            # Only create and save plots when in training mode
            # Create a new figure for rewards plot
            plt.figure(figsize=(12, 5))

            # Plot rewards
            plt.subplot(1, 2, 1)
            plt.plot(rewards_per_episode, alpha=0.6, label='Per Episode')

            # Calculate and plot mean rewards
            mean_rewards = np.zeros(episodes)
            for t in range(episodes):
                mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
            plt.plot(mean_rewards, 'r-', linewidth=2, label='Moving Average (100)')

            plt.title('Rewards over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Plot steps per episode
            plt.subplot(1, 2, 2)
            plt.plot(steps_per_episode, alpha=0.6, label='Steps per Episode')

            # Calculate and plot mean steps
            mean_steps = np.zeros(episodes)
            for t in range(episodes):
                mean_steps[t] = np.mean(steps_per_episode[max(0, t-100):(t+1)])
            plt.plot(mean_steps, 'g-', linewidth=2, label='Moving Average (100)')

            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig('mountain_car_results.png')
            print("Performance plots saved as 'mountain_car_results.png'")
            plt.close()

        except Exception as e:
            print(f"Error during plotting: {e}")
            # Alternative saving method if there's an issue with the plot
            np.save('rewards_per_episode.npy', rewards_per_episode)
            np.save('steps_per_episode.npy', steps_per_episode)
            print("Raw data saved as NumPy files instead")

    return q, rewards_per_episode, steps_per_episode

if __name__ == '__main__':
    # Run training
    # q, rewards, steps = run(5000, is_training=True, render=False)

    # Uncomment to run a demonstration after training
    run(1, is_training=False, render=True)
