#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
dp - value iteration 示例
走迷宫
"""

import numpy as np
from Reinforcement.Environments.grid_world import GridworldEnv


def value_iteration():
    """
    进行训练
    """
    V = np.zeros(env.nS)
    optimal_v = optimal_value_function(V)
    policy = optimal_policy_extraction(optimal_v)
    return policy, optimal_v


def optimal_value_function(V):
    """
    计算最优方案
    """
    while True:
        delta = 0
        for s in range(env.nS):
            Q_sa = np.zeros(env.nA)
            for a in range(env.nA):
                for prob_s, next_state, reward, _ in env.P[s][a]:
                    Q_sa[a] += prob_s * (reward + gamma * V[next_state])
            max_value_function_s = np.max(Q_sa)
            delta = max(delta, np.abs(max_value_function_s - V[s]))
            V[s] = max_value_function_s
        if delta < 0.00001:
            break
    return V


def optimal_policy_extraction(V):
    """
    获取规则
    """
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * V[next_state])
        best_action = np.argmax(Q_sa)
        policy[s] = np.eye(env.nA)[best_action]
    return policy


if __name__ == "__main__":
    gamma = 1.0
    env = GridworldEnv()

    final_policy,final_v = value_iteration()

    print("Final Policy ")
    print(final_policy)

    print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
    print(np.reshape(np.argmax(final_policy, axis=1), env.shape))

    print("Final Value Function grid")
    print(final_v.reshape(env.shape))