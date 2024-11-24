import numpy as np

import ActionValueFunction as avf

class dqn:

    #env : gymnasium environment to play in
    def __init__(self, env):
        self.env = env
        self.Q = avf.ActionValueFunction()
        
    
    def train(self, episodes=1000, # Number of episodes to train
              episode_duration=1000, # Number of steps per episode
              epsilon=(lambda x: 0.1), # Epsilon greedy policy (as a function)
              gamma = 0.99, # Discount factor
              feature_representation=(lambda x: x)): # Feature representation of the state

        D = [] # Replay buffer

        for episode in range(episodes):
            # Reset the environment
            state, _ = self.env.reset()
            f_state = feature_representation(state)
            
            done = False

            for t in range(episode_duration):
                if done:
                    break

                if np.random.rand() < epsilon(episode):
                    at = self.env.action_space.sample()
                else:
                    at = self.Q.best_action(f_state, self.env.action_space)

                next_state, reward, done, _, _ = self.env.step(at)
                f_next_state = feature_representation(next_state)

                D.append((f_state, at, reward, f_next_state))
                
                # Sample a random minibatch of D 
                random_index = np.random.randint(0, len(D))
                minibatch = D[random_index]

                if done:
                    target = reward
                else:
                    target = reward + gamma*self.Q.best_qvalue(f_next_state, self.env.action_space)
                
                self.Q.gradient_descent(target, minibatch)

                #for next iteration
                f_state = f_next_state

