#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Temporal difference = Monte Carlo + Dynamic Programming
Temporal Difference Control - SARSA - on policy
从给定状态，计算得到可以使收益最大化的策略
"""

import numpy as np
from collections import defaultdict
from Reinforcement.Environments.windy_gridworld import WindyGridworldEnv


def sarsa(env, gamma, alpha, nA, epsilon, total_episodes):
    """
    训练
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for k in range(total_episodes):
        current_state = env.reset()
        prob_scores = get_epision_greedy_action_policy(nA, epsilon, Q, current_state)
        current_action = np.random.choice(np.arange(nA), p=prob_scores)

        while True:
            next_state, reward, done, _ = env.step(current_action)

            prob_scores_next_state = get_epision_greedy_action_policy(nA, epsilon, Q, next_state)
            next_action = np.random.choice(np.arange(nA), p=prob_scores_next_state)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[current_state][current_action]
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * td_error

            if done:
                break

            current_state = next_state
            current_action = next_action
    return Q


def get_epision_greedy_action_policy(nA, epsilon, Q, observation):
    """
    评价函数
    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A


if __name__ == "__main__":
    env = WindyGridworldEnv()
    Q = sarsa(env, gamma=1.0, alpha=0.1, nA=env.action_space.n, epsilon=0.1, total_episodes=100)
    print(Q)