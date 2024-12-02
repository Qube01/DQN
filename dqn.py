import numpy as np
import tensorflow as tf
import pickle as pkl
from collections import deque
import ActionValueFunction as avf

class dqn:
    #env : gymnasium environment to play in
    def __init__(self, env, state_size, learning_rate=0.01, Q=None):
        self.env = env
        self.action_space = list(range(env.action_space.n))

        if Q is not None:
            self.Q = Q
        else:
            self.Q = avf.ActionValueFunction(state_size, self.action_space, learning_rate)
            
    def train(self, episodes=1000, # Number of episodes to train
              episode_duration=1000, # Number of steps per episode
              epsilon=(lambda x: 0.1), # Epsilon greedy policy (as a function)
              gamma = 0.99, # Discount factor
              feature_representation=(lambda x: x)): # Feature representation of the state

        D = deque(maxlen=10000)  # Replay buffer

        for episode in range(episodes):
            # Reset the environment
            state, _ = self.env.reset()
            fstate = feature_representation(state)
            
            done = False

            loss_values = []

            for t in range(episode_duration):
                if done:
                    break

                if np.random.rand() < epsilon(episode):
                    at = np.random.choice(self.action_space)
                else:
                    at = self.Q.get_best_action(fstate, self.action_space)
                    #print("the best action is ", at)

                next_state, reward, done, _, _ = self.env.step(at)
                next_fstate = feature_representation(next_state)

                D.append((fstate, at, reward, next_fstate, done))
                
                if len(D) >= 32:  # Start training after filling the replay buffer
                    minibatch = np.random.choice(len(D), 32, replace=False)
                    fstates, actions, rewards, next_fstates, dones = zip(*[D[idx] for idx in minibatch])
                    targets = []
                    for r_j, fs_next_j, done_j in zip(rewards, next_fstates, dones):
                        if done_j:
                            targets.append(r_j)
                        else:
                            reward_tensor = tf.constant(r_j, dtype=tf.float32)
                            gamma_tensor = tf.constant(gamma, dtype=tf.float32)
                            best_qvalue = self.Q.get_best_qvalue(fs_next_j, self.action_space)
                            targets.append(reward_tensor + gamma_tensor * best_qvalue)
                    targets = tf.convert_to_tensor(targets, dtype=tf.float32)
                    loss_values.append(self.Q.train_step(targets, fstates, actions))

                #for next iteration
                fstate = next_fstate

            print(f"Episode {episode}: Average loss = {np.mean(loss_values)}")
        
        # Save the trained model
        with open('trained_model.pkl', 'wb') as f:
            pkl.dump(self.Q, f)
