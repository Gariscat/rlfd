from tqdm import trange
import numpy as np


def default_L1(target, pred):
    dis = np.abs(pred-target).item()
    return np.exp(-dis)


def evaluate(env, model, num_episodes=100):
    criteria = {
        'step_rewards': [],
        'episode_rewards': [],
        'avg_step_reward': 0,
        'avg_episode_reward': 0,
    }
    for i_episode in trange(num_episodes, desc='Evaluation'):
        obs = env.reset()
        done = False
        r = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            criteria['step_rewards'] += [reward]
            r += reward

        criteria['episode_rewards'] += [r]

    criteria['avg_step_reward'] = np.mean(criteria['step_rewards'])
    criteria['avg_episode_reward'] = np.mean(criteria['episode_rewards'])

    return criteria
