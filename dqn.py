import numpy as np
import tensorflow as tf
import ActionValueFunction as avf

class dqn:
    #env : gymnasium environment to play in
    def __init__(self, env, theta_size, learning_rate=0.01):
        self.env = env
        self.Q = avf.ActionValueFunction(theta_size, learning_rate)
        self.action_space = list(range(env.action_space.n))

    def train(self, episodes=1000, # Number of episodes to train
              episode_duration=1000, # Number of steps per episode
              epsilon=(lambda x: 0.1), # Epsilon greedy policy (as a function)
              gamma = 0.99, # Discount factor
              feature_representation=(lambda x: x)): # Feature representation of the state

        D = [] # Replay buffer

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
                    at = self.action_space.sample()
                else:
                    at = self.Q.get_best_action(fstate, self.action_space)

                print("here")
                next_state, reward, done, _, _ = self.env.step(at)
                next_fstate = feature_representation(next_state)

                D.append((fstate, at, reward, next_fstate, done))
                
                # Sample a random minibatch of D 
                random_index = np.random.randint(0, len(D))
                fs_j, a_j, r_j, fs_next_j, done_j = D[random_index]

                if done_j:
                    target = tf.constant(r_j, dtype=tf.float32)  # Convert reward to a TensorFlow scalar
                else:
                    reward_tensor = tf.constant(r_j, dtype=tf.float32)
                    gamma_tensor = tf.constant(gamma, dtype=tf.float32)
                    best_qvalue = self.Q.get_best_qvalue(fs_next_j, self.action_space)
                    target = reward_tensor + gamma_tensor * best_qvalue  # TensorFlow operations
                
                loss_values.append(self.Q.train_step(target, fs_j, a_j))

                #for next iteration
                fstate = next_fstate

            print(f"Episode {episode}: Average loss = {np.mean(loss_values)}")
