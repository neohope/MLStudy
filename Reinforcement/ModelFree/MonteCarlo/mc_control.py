#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Monte Carlo Control
从给定状态，计算得到可以使收益最大化的策略
"""

import numpy as np
from collections import defaultdict
from Reinforcement.Environments.black_jack import BlackjackEnv


def mc_control_epsilon_greedy(env, epsilon, nA, total_episodes):
    """
    训练
    """
    returns_sum = defaultdict(float)
    states_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for k in range(total_episodes):
        episode = generate_episode(env, epsilon, nA, Q)
        state_actions_in_episode = list(set([(sar[0], sar[1]) for sar in episode]))
        for i, sa_pair in enumerate(state_actions_in_episode):
            state, action = sa_pair
            G = sum([sar[2] for i, sar in enumerate(episode[i:])])
            returns_sum[sa_pair] += G
            states_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / states_count[sa_pair]
    return Q


def generate_episode(env, epsilon, nA, Q):
    """
    要牌，一直到游戏结束
    """
    episode = []
    current_state = env.reset()
    while (True):
        prob_scores = get_epision_greedy_action_policy(epsilon, nA, Q, current_state)
        action = np.random.choice(np.arange(len(prob_scores)), p=prob_scores)  # 0 or 1
        next_state, reward, done, _ = env.step(action)
        episode.append((current_state, action, reward))
        if done:
            break
        current_state = next_state
    return episode


def get_epision_greedy_action_policy(epsilon, nA, Q, observation):
    """
    评价函数
    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A


if __name__ == "__main__":
    env = BlackjackEnv()
    Q = mc_control_epsilon_greedy(env, epsilon=0.1, nA=2, total_episodes=50000)
    print(Q)