#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CNN/Convnets/Convolutional neural networks
keras tensorflow
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.optimizers import SGD


def train():
    """
    训练
    """
    epochs = 10
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)
    return model


def create_model():
    """
    创建模型
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu",input_shape=(3, 32, 32), padding = "same", kernel_constraint=maxnorm(3)))
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(3, 32, 32), padding="same", kernel_constraint=maxnorm(3)))
    # 防止过拟合
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(10, activation='softmax'))
    return model


if __name__ == '__main__':
    # 加载测试数据，一堆32*32*3的图片
    # 10类
    # 50000训练数据
    # 10000测试数据
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    print(X_train.shape)
    print(X_test.shape)

    # 将数据转换为0-1的浮点数
    X_train=X_train/255.0
    X_test=X_test/255.0

    # 将Y转换为标签矩阵
    # 属于哪一类，哪一列为1，其余为0
    Y_train=np_utils.to_categorical(Y_train)
    Y_test=np_utils.to_categorical(Y_test)

    # reshape for tf
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # 训练
    model = train()

    # 模型准群率评估
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Final Accuracy: %.2f%%" % (scores[1]*100))

    #保存模型及训练结果
    jsonFile=model.to_json()
    with open('output/cifar10.json','w') as file:
        file.write(jsonFile)
    model.save_weights('output/cifar10.h5')
