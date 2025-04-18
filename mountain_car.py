import gymnasium as gym
import time

def main():
    # Create the Mountain Car Continuous environment with human rendering mode
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    # Reset the environment
    state, _ = env.reset(seed=0)

    # Run simulation with constant acceleration
    done = False
    truncated = False
    step = 0

    # You can change this value to control the car's movement
    # 0.0 = no force, 1.0 = maximum positive force
    action_value = 1.0

    print("Running Mountain Car Continuous environment...")
    print("Press Ctrl+C to stop")

    try:
        while not (done or truncated):
            # Apply the action
            action = np.array([action_value])

            # Step the environment (this will also render it)
            state, reward, done, truncated, _ = env.step(action)

            # Optional: slow down the simulation to make it more visible
            time.sleep(0.01)

            step += 1

            # Reset if done
            if done or truncated:
                print(f"Episode finished after {step} steps")
                state, _ = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("Simulation stopped by user")

    # Clean up
    env.close()

if __name__ == "__main__":
    import numpy as np
    main()
