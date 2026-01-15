import sheepshead_v1
from pettingzoo.test import api_test, performance_benchmark

"""
if __name__ == "__main__":
    my_env = sheepshead_v1.env()
    #api_test(my_env, verbose_progress=True)
    performance_benchmark(my_env)

"""
env = sheepshead_v1.env()
env.reset(seed=42)
while len(env.agents) > 0:
    agent = env.agent_selection
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)
    print(agent, action)
    env.step(action)

# https://pettingzoo.farama.org/tutorials/sb3/connect_four/