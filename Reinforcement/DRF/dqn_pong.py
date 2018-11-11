#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
训练程序玩Atari Pong游戏
Deep QLearning

最好在linux下运行
需要安装软件包
pip install gym[atari]
"""

import numpy as np
import pickle
import gym


def init(resume):
    """
    初始化
    """
    H = 200      # number of hidden layer neurons
    D = 80 * 80  # input dimensionality: 80x80 grid

    if resume:
        model = pickle.load(open('pong.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H, D) / np.sqrt(D)
        model['W2'] = np.random.randn(H) / np.sqrt(H)

    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory
    env = gym.make("Pong-v0")
    return D,model,grad_buffer,rmsprop_cache,env


def train(D, model, grad_buffer, rmsprop_cache, env, gamma, batch_size, learning_rate, decay_rate):
    """
    训练
    gamma          # discount factor for reward
    batch_size     # every how many episodes to do a param update?
    learning_rate
    decay_rate     # decay factor for RMSProp leaky sum of grad^2
    """

    render = True
    reward_sum = 0
    episode_number = 0
    prev_x = None  # used in computing the difference frame
    running_reward = None
    xs, hs, dlogps, drs = [], [], [], []
    observation = env.reset()

    while True:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(model, x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(gamma, epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(model, epx, eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('pong.p', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward))
            print('' if reward == -1 else ' !!!!!!!!')


def prepro(I):
    """
    将图形帧转换为向量
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(gamma, r):
    """
    计算奖励
    take 1D float array of rewards and compute discounted reward
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    """
    前向传递
    """
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def policy_backward(model, epx, eph, epdlogp):
    """
    后向传递
    backward pass.
    eph is array of intermediate hidden states
    """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


if __name__ == "__main__":
    # 从保存点加载游戏
    resume = False
    # 初始化
    D, model, grad_buffer,rmsprop_cache, env=init(resume)
    # 训练
    train(D, model, grad_buffer,rmsprop_cache, env, gamma = 0.99, batch_size = 10, learning_rate = 1e-4, decay_rate = 0.99 )
