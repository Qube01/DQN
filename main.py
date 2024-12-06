import gymnasium as gym
import torch
import ActionValueFunction as avf
import dqn

env_text = "LunarLander-v3"

def train():
    env = gym.make(env_text, continuous=False, render_mode="human")
    DQN = dqn.dqn(env, state_size=8, learning_rate=0.001)
    DQN.train(episodes=2000000,
              episode_duration=500,
              epsilon=(lambda x: max(1 - (x / 2000000), 0.1)),
              feature_representation=(lambda x: x))

def run():
    env = gym.make(env_text, continuous=False, render_mode="human")
    Q = avf.ActionValueFunction(state_size=8, action_space=list(range(env.action_space.n)))  # Create a new ActionValueFunction instance
    Q.model.load_state_dict(torch.load('trained_model.pth'))  # Load the saved weights
    Q.model.eval()  # Set the model to evaluation mode
    DQN = dqn.dqn(env, state_size=8, Q=Q)  # Create dqn with the loaded Q

    state, _ = env.reset()
    fstate = state
    done = False

    while not done:
        env.render()
        action = DQN.Q.get_best_action(fstate, DQN.action_space)
        next_state, _, done, _, _ = env.step(action)
        fstate = next_state

train()