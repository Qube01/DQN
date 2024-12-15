import gymnasium as gym
import tensorflow as tf
import ActionValueFunction as avf
import dqn

env_text = "LunarLander-v3"


def train():
    env = gym.make(env_text, continuous=False, render_mode="rgb_array")
    DQN = dqn.dqn(env, state_size=8, learning_rate=0.001)
    y=[2.5, 2.5, 10, 10, 7, 10, 1, 1]
    DQN.train(episodes=1000,
              episode_duration=500,
              epsilon=(lambda x: max(1 - (x / 500), 0.1)),
              feature_representation=(lambda x: x))

def run():
    env = gym.make(env_text, continuous=False, render_mode="human")
    # Load the model weights instead of the entire object
    Q = avf.ActionValueFunction(state_size=8, action_space=list(range(env.action_space.n)))  # Create a new ActionValueFunction instance
    Q.model = tf.keras.models.load_model('model_tf/trained_model_episode_900.keras')  # Load the saved weights
    DQN = dqn.dqn(env, state_size=8, learning_rate=0.01, Q=Q)  # Create dqn with the loaded Q

    state, _ = env.reset()
    fstate = state
    done = False

    reward=0

    while not done:
        env.render()
        action = DQN.Q.get_best_action(fstate, DQN.action_space)
        next_state, r, done, _, _ = env.step(action)
        reward+=r
        fstate = next_state

    print(f"Total reward: {reward}")

#train()
run()