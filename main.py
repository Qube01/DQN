import gymnasium as gym
import dqn as dqn
import pickle as pkl

env_text = "LunarLander-v3"


def train():
    env = gym.make(env_text, continuous=False, render_mode="human")
    DQN = dqn.dqn(env, state_size=8, learning_rate=0.1)
    DQN.train(episodes=100, 
              episode_duration=500, 
              epsilon=(lambda x: 0))#1 - (x / 1000)))

def run():
    env = gym.make(env_text, continuous=False, render_mode="human")
    with open("trained_model.pkl", "rb") as file:
        Q = pkl.load(file)
        DQN = dqn.dqn(env, state_size=8, learning_rate=0.01, Q=Q)
    
    state, _ = env.reset()
    fstate = state
    done = False

    while not done:
        env.render()
        action = DQN.Q.get_best_action(fstate, DQN.action_space)
        next_state, _, done, _, _ = env.step(action)
        fstate = next_state

train()
#run()