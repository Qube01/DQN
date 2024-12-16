import os
import torch
import numpy as np
from collections import deque
import ActionValueFunction as avf

class dqn:
    def __init__(self, env, state_size, learning_rate=0.01):
        self.env = env
        self.action_space = list(range(env.action_space.n))

        self.Q = avf.ActionValueFunction(state_size, self.action_space, learning_rate)


    def train(self, episodes=1000, episode_duration=1000, epsilon=(lambda x: 0.1), gamma=0.99, feature_representation=(lambda x: x)):
        D = deque(maxlen=100000)
        save_dir = 'model_torch'
        os.makedirs(save_dir, exist_ok=True)

        for episode in range(episodes):
            state, _ = self.env.reset()
            fstate = feature_representation(state)

            done = False
            loss_values = []

            total_reward = 0
            for t in range(episode_duration):

                if done:
                    break

                if np.random.rand() < epsilon(episode):
                    at = np.random.choice(self.action_space)
                else:
                    at = self.Q.get_best_action(fstate, self.action_space)

                next_state, reward, terminated, truncated, _ = self.env.step(at)
                next_fstate = feature_representation(next_state)

                total_reward += reward
                done = terminated or truncated
                
                D.append((fstate, at, reward, next_fstate, done))

                minibatch_size = 32

                if len(D) >= minibatch_size:
                    minibatch = np.random.choice(len(D), minibatch_size, replace=False)
                    fstates, actions, rewards, next_fstates, dones = zip(*[D[idx] for idx in minibatch])
                    targets = []
                    for r_j, fs_next_j, done_j in zip(rewards, next_fstates, dones):
                        if done_j:
                            targets.append(float(r_j))
                        else:
                            best_qvalue = self.Q.get_best_qvalue(fs_next_j, self.action_space)
                            targets.append(float(r_j) + gamma * float(best_qvalue))
                    targets = torch.tensor(targets, dtype=torch.float32)
                    loss_values.append(self.Q.train_step(targets, fstates, actions))

                fstate = next_fstate

            if (episode + 1) % 10 == 0:
                try:
                    torch.save(self.Q.model.state_dict(), os.path.join(save_dir, f'last_trained_model.pth'))
                except Exception as e:
                    print(f"Error saving model at episode {episode + 1}: {e}")


            print(f"\n Episode {episode}: Average loss = {np.mean(loss_values)}")

            if total_reward >= 0:
                print(f"\033[92mEpisode {episode}: Total reward = {total_reward}\033[0m")
            else:
                print(f"\033[91mEpisode {episode}: Total reward = {total_reward}\033[0m")

            print()
            print(f"Deque size: {len(D)}")

        try:
            torch.save(self.Q.model.state_dict(), os.path.join(save_dir, 'trained_model.pth'))
        except Exception as e:
            print(f"Error saving final model: {e}")