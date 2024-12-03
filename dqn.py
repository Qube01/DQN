import numpy as np
import tensorflow as tf
import pickle as pkl
from collections import deque
import ActionValueFunction as avf

class dqn:
    def __init__(self, env, state_size, learning_rate=0.01, Q=None):
        self.env = env
        self.action_space = list(range(env.action_space.n))

        if Q is not None:
            self.Q = Q
        else:
            self.Q = avf.ActionValueFunction(state_size, self.action_space, learning_rate)
            
    def train(self, episodes=1000, episode_duration=1000, epsilon=(lambda x: 0.1), gamma=0.99, feature_representation=(lambda x: x)):
        D = deque(maxlen=10000)

        for episode in range(episodes):
            state, _ = self.env.reset()
            fstate = feature_representation(state)
            
            done = False
            loss_values = []

            total_reward=0
            for t in range(episode_duration):
                if done:
                    break

                if np.random.rand() < epsilon(episode):
                    at = np.random.choice(self.action_space)
                else:
                    at = self.Q.get_best_action(fstate, self.action_space)

                next_state, reward, done, _, _ = self.env.step(at)
                next_fstate = feature_representation(next_state)

                total_reward+=reward

                D.append((fstate, at, reward, next_fstate, done))
                
                if len(D) >= 32:
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

                fstate = next_fstate

            print(f"Episode {episode}: Total reward = {total_reward}")
            print(f"Episode {episode}: Average loss = {np.mean(loss_values)}")
        
        with open('trained_model.pkl', 'wb') as f:
            pkl.dump(self.Q, f)
