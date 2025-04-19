import gymnasium as gym
import time
import torch
import torch.nn as nn
import numpy as np

class Sign(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(Sign, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.sign(x + self.epsilon)

class Step(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(Step, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Returns 0 for negative values, 1 for non-negative values
        return (x + self.epsilon >= 0).float()

class MountainCarMLP(nn.Module):
    def __init__(self):
        super(MountainCarMLP, self).__init__()

        # Define a sequential model for easier addition of layers
        self.model = nn.Sequential(
            nn.Linear(2, 2),    # First hidden layer: decision boundaries
            Step(),
            nn.Linear(2, 2),    # Second hidden layer: xor behaviour
            nn.ReLU(),
            nn.Linear(2, 1),    # Output layer: xor behaviour for moving left or right
            Sign(),
        )

        # Initialize weights for the first layer
        with torch.no_grad():
            # First hidden layer:
            # Decision boundary at valley, and movement right or left?
            self.model[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            self.model[0].bias.data = torch.tensor([-0.5, 0.0])

            # Second hidden layer: xor behaviour
            self.model[2].weight.data = torch.ones(2, 2)
            self.model[2].bias.data = torch.tensor([0.0, -1.0])

            # Output layer: xor behaviour, maps from -0.5 to 0.5
            self.model[4].weight.data = torch.tensor([[1.0, -2.0]])
            self.model[4].bias.data = torch.tensor([-0.5])

    def forward(self, x):
        return self.model(x)


def main():
    # Create the Mountain Car Continuous environment with human rendering mode
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    # Initialize our PyTorch neural network
    model = MountainCarMLP()
    model.eval()  # Set to evaluation mode

    # Reset the environment
    state, _ = env.reset(seed=0)

    # Run simulation with neural network control
    done = False
    truncated = False
    step = 0

    print("Running Mountain Car Continuous environment with PyTorch Neural Network...")
    print("Press Ctrl+C to stop")

    try:
        while not (done or truncated):
            # Convert state to torch tensor
            state_tensor = torch.FloatTensor(state)

            # Get action from neural network
            with torch.no_grad():
                action = model(state_tensor).numpy().astype(np.float32)

            # Step the environment (this will also render it)
            state, reward, done, truncated, _ = env.step(action)

            # Print some information
            position, velocity = state
            print(f"Step: {step}, Position: {position:.4f}, Velocity: {velocity:.4f}, Action: {action[0]:.4f}")

            # Optional: slow down the simulation to make it more visible
            time.sleep(0.01)

            step += 1

            # Reset if done or truncated
            if done or truncated:
                print(f"Episode finished after {step} steps")
                state, _ = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("Simulation stopped by user")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()