from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re


def default_L1(target, pred):
    dis = np.abs(pred-target).item()
    return np.exp(-dis)


def plot_trace(
    targ_trace: list,
    pred_trace: list,
    figsize=(100, 5),
    save_path='runs/'+re.sub(r'\W+', '', str(datetime.now()))+'.jpg',
    scale=1,
    debug=False,
    right_lim=10,
):
    horizon = max(targ_trace+pred_trace)
    plt.figure(figsize=figsize)
    plt.axhline(y=0.5, xmin=0, xmax=horizon, color='black', linestyle=':')
    for targ, pred in zip(targ_trace, pred_trace):
        plt.axvline(x=targ*scale, ymin=0, ymax=0.5, color='b')
        # predictions
        plt.axvline(x=pred*scale, ymin=0.5, ymax=1, color='r')

    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    if save_path:
        plt.savefig(save_path)
    if debug:
        plt.show()
    plt.close()


def evaluate(env, model, num_episodes=100, plot=False):
    criteria = {
        'step_rewards': [],
        'episode_rewards': [],
        'avg_step_reward': 0,
        'avg_episode_reward': 0,
    }
    for i_episode in trange(num_episodes, desc='Evaluation'):
        targ_trace, pred_trace = [], []

        obs = env.reset()
        done = False
        r = 0

        cur = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            criteria['step_rewards'] += [reward]
            r += reward

            target = env.state[-1, 0]
            targ_trace += [target+cur]
            pred_trace += [action+cur]
            cur += target

        # print(pred_trace)
        # print(targ_trace)
        # print(len(pred_trace), len(targ_trace))

        criteria['episode_rewards'] += [r]
        if plot:
            plot_trace(targ_trace, pred_trace, scale=env._scale)

    criteria['avg_step_reward'] = np.mean(criteria['step_rewards'])
    criteria['avg_episode_reward'] = np.mean(criteria['episode_rewards'])

    return criteria
