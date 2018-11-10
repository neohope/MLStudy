#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
dp - policy iteration 示例
走迷宫
"""

import numpy as np
from Reinforcement.Environments.grid_world import GridworldEnv


def policy_iteration(env, gamma):
    """
    进行训练
    """
    # 从一个随机策略开始
    policy = np.ones([env.nS, env.nA]) / env.nA

    epochs = 1000
    for i in range(epochs):
        V = policy_evaluation(env, gamma, policy)
        old_policy = np.copy(policy)
        new_policy = policy_improvement(env, gamma, V, old_policy)
        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy

    return policy, V


def policy_evaluation(env, gamma, policy):
    """
    评估policy
    """
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # 棋盘
        for s in range(env.nS):
            total_state_value = 0
            # 四个方向，或者是四个Action
            for a, prob_a in enumerate(policy[s]):
                # 计算奖励
                for prob_s, next_state, reward, _ in env.P[s][a]:
                    total_state_value += prob_a * prob_s * (reward + gamma * V[next_state])
            # 计算变化
            delta = max(delta, np.abs(total_state_value - V[s]))
            V[s] = total_state_value

        # 如果变化很小，退出循环
        if delta < 0.005:
            break

    return np.array(V)


def policy_improvement(env, gamma, V, policy):
    """
    重新计算policy
    """
    for s in range(env.nS):
        Q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * V[next_state])
        best_action = np.argmax(Q_sa)
        policy[s] = np.eye(env.nA)[best_action]

    return policy


if __name__ == "__main__":
    gamma =1.0
    env = GridworldEnv()

    final_policy, final_v = policy_iteration(env, gamma)

    print("Final Policy ")
    print(final_policy)

    print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
    print(np.reshape(np.argmax(final_policy, axis=1), env.shape))

    print("Final Value Function grid")
    print(final_v.reshape(env.shape))
