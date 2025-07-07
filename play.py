import gymnasium as gym
from reinforce import Net, Agent


env = gym.make("CartPole-v1", render_mode="rgb_array")

in_features  = env.observation_space.shape[0]
out_features = env.action_space.n

net = Net(in_features=in_features, out_features=out_features)
net.train()

agent = Agent(net, env)
agent.learn(500)
