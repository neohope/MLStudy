#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Temporal Difference Prediction
从给定状态，计算得到可以使收益最大化的策略
"""

import numpy as np
from collections import defaultdict
from Reinforcement.Environments.black_jack import BlackjackEnv


def td_prediction(env, gamma, alpha, nA, total_episodes):
    """
    训练
    """
    V = defaultdict(float)
    for k in range(1, total_episodes + 1):
        current_state = env.reset()
        while True:
            prob_scores = get_action_policy(current_state)
            current_action = np.random.choice(np.arange(nA), p=prob_scores)
            next_state, reward, done, _ = env.step(current_action)
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[current_state]
            V[current_state] = V[current_state] + alpha * td_error
            if done:
                break
            current_state = next_state
    return V


def get_action_policy(state):
    """
    评价函数
    """
    score, dealer_score, usable_ace = state
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])


if __name__ == "__main__":
    env = BlackjackEnv()
    V = td_prediction(env, gamma=1.0, alpha=0.5, nA=2, total_episodes=100000)
    print(V)