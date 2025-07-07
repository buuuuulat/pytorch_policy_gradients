import gymnasium as gym
from reinforce import Net, Agent


env = gym.make("CartPole-v1")

in_features  = env.observation_space.shape[0]
out_features = env.action_space.n

net = Net(in_features=in_features, out_features=out_features)

agent = Agent(net, env)
agent.learn(1500)


eval_env = gym.make("CartPole-v1", render_mode="human")
eval_agent = Agent(net, eval_env)
score = eval_agent.play(render=True)
print("Play-out total reward:", score)

# Закрываем среду, чтобы окно точно закрылось
eval_env.close()