#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
训练程序玩Atari打砖块游戏

最好在linux下运行
需要安装软件包
pip install gym[atari]
"""

import gym
import pickle
import random
import warnings
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop


ACTION_SIZE = 3
ATARI_SHAPE = (84, 84, 4)


def atari_model():
    """
    模型初始化
    """
    # actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
    # frames_input = layers.Input(ATARI_SHAPE, name='inputs')
    actions_input = layers.Input((ACTION_SIZE,), name='mask')
    frames_input = layers.Input(ATARI_SHAPE, name='img')

    normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(frames_input)
    conv_1 = layers.convolutional.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    conv_2 = layers.convolutional.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_flattened = layers.core.Flatten()(conv_2)
    hidden = layers.Dense(256, activation='relu')(conv_flattened)

    output = layers.Dense(ACTION_SIZE)(hidden)
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


def huber_loss(y, q_value):
    """
    损失函数
    """
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


def train(env, model, target_model, gamma, epslion, final_epsilon, nEpisodes, batch_size, total_observe_count, target_model_change):
    """
    训练程序玩Atari打砖块游戏
    """
    render = True
    max_score = 0
    epsilon_step_num = 100000
    replay_memory = deque(maxlen=400000)
    epsilon_decay = (1.0 - final_epsilon)/epsilon_step_num
    state = env.reset()

    for episode in range(nEpisodes):
        if render: env.render()

        dead, done, lives_remaining, score = False, False, 5, 0

        current_state = env.reset()
        for _ in range(random.randint(1, 30)):
            current_state, _, _, _ = env.step(1)

        current_state = pre_process(current_state)
        current_state = np.stack((current_state, current_state, current_state, current_state), axis=2)
        current_state = np.reshape([current_state], (1, 84, 84, 4))

        while not done:

            action = epslion_greedy_policy_action(current_state, epslion, episode, total_observe_count)
            real_action = action + 1

            if epslion > final_epsilon and episode > total_observe_count:
                epslion -= epsilon_decay

            next_state, reward, done, lives_left = env.step(real_action)

            next_state = pre_process(next_state)  # 84,84 frame
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_state = np.append(next_state, current_state[:, :, :, :3], axis=3)

            if lives_remaining > lives_left['ale.lives']:
                dead = True
                lives_remaining = lives_left['ale.lives']

            replay_memory.append((current_state, action, reward, next_state, dead))

            if episode > total_observe_count:
                deepQlearn(replay_memory, batch_size, gamma)

                if episode % target_model_change == 0:
                    target_model.set_weights(model.get_weights())

            score += reward

            if dead:
                dead = False
            else:
                current_state = next_state

            if max_score < score:
                print("max score for the episode {} is : {} ".format(episode, score))
                max_score = score

        if episode % 100 == 0:
            print("final score for the episode {} is : {} ".format(episode, score))
            save_model(model)


def pre_process(frame_array):
    """
    训练程序玩Atari打砖块游戏
    """
    # 转为灰阶图像
    grayscale_frame = rgb2gray(frame_array)
    # 调整图像大小
    resized_frame = np.uint8(resize(grayscale_frame, (84, 84), mode='constant') * 255)
    return resized_frame


def epslion_greedy_policy_action(current_state, epslion, episode, total_observe_count):
    """
    选择Action
    """
    if np.random.rand()<=epslion or episode<total_observe_count:
        #随机Action
        return random.randrange(ACTION_SIZE)
    else:
        #最优Action
        Q_value = model.predict([current_state, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(Q_value[0])


def deepQlearn(replay_memory, batch_size, gamma):
    """
    Deep QLearn
    """
    current_state_batch, actions, rewards, next_state_batch, dead = get_sample_random_batch_from_replay_memory(replay_memory, batch_size)
    actions_mask = np.ones((batch_size, ACTION_SIZE))
    next_Q_values = target_model.predict([next_state_batch, actions_mask])  # separate old model to predict
    targets = np.zeros((batch_size,))

    for i in range(batch_size):
        if dead[i]:
            targets[i] = -1
        else:
            targets[i] = rewards[i] + gamma * np.amax(next_Q_values[i])

    one_hot_actions = np.eye(ACTION_SIZE)[np.array(actions).reshape(-1)]
    one_hot_targets = one_hot_actions * targets[:, None]
    model.fit([current_state_batch, one_hot_actions], one_hot_targets, epochs=1, batch_size=batch_size, verbose=0)


def get_sample_random_batch_from_replay_memory(replay_memory, batch_size):
    """
    获取随机操作列表
    """
    mini_batch = random.sample(replay_memory, batch_size)
    current_state_batch = np.zeros((batch_size, 84, 84, 4))
    next_state_batch = np.zeros((batch_size, 84, 84, 4))

    actions, rewards, dead = [], [], []
    for idx, val in enumerate(mini_batch):
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]
        dead.append(val[4])

    return current_state_batch, actions, rewards, next_state_batch, dead


def save_model(model):
    """
    保存游戏
    """
    model_name = "BreakoutDeterministic.p"
    model.save(model_name)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    env = gym.make('BreakoutDeterministic-v4')

    resume = False
    if resume:
        model = pickle.load(open('BreakoutDeterministic.p', 'rb'))
    else :
        model = atari_model()

    target_model = atari_model()
    train(env, model, target_model, gamma=0.99, epslion=1.0, final_epsilon=0.1, nEpisodes=100000, batch_size=32, total_observe_count=750, target_model_change=100)
