import gym


step = 0

env = gym.make('Breakout-v0')

# env.action_space

# env = gym.make("Taxi-v2")
# env = gym.make('CartPole-v0')

# observation = env.reset()
env.reset()

for _ in range(step):
  # for visualization
  env.render()

  action = env.action_space.sample() # action: int number
  # your agent here (this takes random actions)

  # env.step(env.action_space.sample())
  observation, reward, done, info = env.step(action)
  # observation(object): pixel data (for games e.g. breakout)
  #     low-dim nparray or int for Taxi and CartPole
  # reward: float type reward
  # done: True or False if the game is done
  # info: a dictionary like how many lives still available

  if done:
    env.reset()
