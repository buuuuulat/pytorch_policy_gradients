import gymnasium as gym
from baseline import Net as BaselineNet, Agent as BaselineAgent
from reinforce import Net as ReinforceNet, Agent as ReinforceAgent


# baseline test
env = gym.make("LunarLander-v3")

in_features  = env.observation_space.shape[0]
out_features = env.action_space.n

net = BaselineNet(in_features=in_features, out_features=out_features)

agent = BaselineAgent(net, env)
agent.learn(5000)


# reinforce test
env = gym.make("LunarLander-v3")

in_features  = env.observation_space.shape[0]
out_features = env.action_space.n

net = ReinforceNet(in_features=in_features, out_features=out_features)

agent = ReinforceAgent(net, env)
agent.learn(5000)


"""
eval_env = gym.make("LunarLander-v3", render_mode="human")
eval_agent = Agent(net, eval_env)
score = eval_agent.play(render=True)
print("Play-out total reward:", score)

eval_env.close()
"""
