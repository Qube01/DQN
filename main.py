import gymnasium as gym
import torch
import ActionValueFunction as avf
import dqn

env_text = "LunarLander-v3"

def train():
    env = gym.make(env_text, continuous=False, render_mode=None)
    DQN = dqn.dqn(env, state_size=8, learning_rate=0.0005)
    DQN.train(episodes=5000,
              episode_duration=5000,
              epsilon=(lambda x: max(1 - (x / 5000), 0.1)),
              feature_representation=(lambda x: x))

def run():
    env = gym.make(env_text, continuous=False, render_mode="human")

    DQN = dqn.dqn(env, state_size=8)  # Create dqn with the loaded Q
    DQN.Q.model.load_state_dict(torch.load('model_torch/2/last_trained_model.pth'))  # Load the saved weights
    DQN.Q.model.eval()
    state, _ = env.reset()
    fstate = state
    done = False

    reward = 0

    while not done:
        env.render()
        action = DQN.Q.get_best_action(fstate, DQN.action_space)
        next_state, r, done, _, _ = env.step(action)
        reward += r
        fstate = next_state

    print(f"Total reward: {reward}")

#train()
run()