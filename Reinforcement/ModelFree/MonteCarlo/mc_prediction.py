#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Monte Carlo Prediction
从给定状态，通过策略，预测整体收益
"""

from collections import defaultdict
from Reinforcement.Environments.black_jack import BlackjackEnv


def mc_prediction_blackjack(env,total_episodes):
    """
    训练 stationary
    """
    returns_sum = defaultdict(float)
    states_count = defaultdict(float)
    V = defaultdict(float)

    for k in range(1, total_episodes + 1):
        episode = generate_episode(env)
        # sar--> state,action,reward
        states_in_episode = list(set([sar[0] for sar in episode]))

        for i, state in enumerate(states_in_episode):
            G = sum([sar[2] for i, sar in enumerate(episode[i:])])
            # for stationary problems
            returns_sum[state] += G
            states_count[state] += 1.0
            V[state] = returns_sum[state] / states_count[state]

    return V


def mc_prediction_non_stationary(env,total_episodes):
    """
    训练 non stationary
    """
    V = defaultdict(float)

    for k in range(1, total_episodes + 1):
        episode = generate_episode(env)
        # sar--> state,action,reward
        states_in_episode = list(set([sar[0] for sar in episode]))

        for i, state in enumerate(states_in_episode):
            G = sum([sar[2] for i, sar in enumerate(episode[i:])])
            # for non stationary problems
            alpha=0.5
            V[state] = V[state]+ alpha*(G-V[state])
    return V


def generate_episode(env):
    """
    要牌，一直到游戏结束
    """
    episode = []
    current_state = env.reset()

    while (True):
        # 0 or 1
        action = get_action_policy(current_state)
        next_state, reward, done, _ = env.step(action)
        episode.append((current_state, action, reward))
        if done:
            break
        current_state = next_state
    return episode


def get_action_policy(state):
    """
    评价函数
    """
    score, dealer_score, usable_ace = state
    return 0 if score >= 20 else 1


if __name__ == "__main__":
    env = BlackjackEnv()
    V = mc_prediction_blackjack(env, 10000)
    print(V)
