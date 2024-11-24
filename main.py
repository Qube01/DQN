import gymnasium as gym
import dqn as dqn

env_text = "LunarLander-v3"
env = gym.make(env_text, continuous=False, render_mode="human")

dqn = dqn.dqn(env, theta_size=9)
dqn.train(episodes=1000, episode_duration=1000)

# # Run the environment
# # env.reset()
# # done = False

# # while not done:
# #     env.render()
# #     action = env.action_space.sample()
# #     _, _, done, _, _ = env.step(action)  

env.close()