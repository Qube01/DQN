import gymnasium as gym

env_text = "LunarLander-v3"
env = gym.make(env_text, continuous=False, render_mode="human")

# Run the environment
env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)  

env.close()