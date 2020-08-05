########################
# 人脸特征训练
########################

import os
import random
import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from load_dataset import load_dataset, resize_image, IMAGE_SIZE, json_labels_read_from_file


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, nb_classes=2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, padding='same', input_shape=(64, 64, 3)))# 1 2维卷积层

        self.model.add(Activation('relu'))  # 2 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层

        self.model.add(Convolution2D(64, 3, 3, padding='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层

        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, train_dataset, test_dataset, batch_size=20, nb_epoch=50, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际模型配置工作

        train_images = []
        train_labels = []
        for image, label in train_dataset:
            print(image.shape)
            train_images.append(image.numpy())
            train_labels.append(label)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        print(train_images.shape)
        print(train_labels.shape)

        valid_images = []
        valid_labels = []
        for image, label in test_dataset:
            valid_images.append(image.numpy())
            valid_labels.append(label)

        valid_images = np.array(valid_images)
        valid_labels = np.array(valid_labels)

        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(train_images,
                           train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(valid_images, valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(train_images, train_labels,
                                                  batch_size=batch_size),
                                     steps_per_epoch=train_images.shape[0] // batch_size,
                                     epochs=nb_epoch,
                                     validation_data=(valid_images, valid_labels))

    MODEL_PATH = './model.tf'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率
        result_probability = self.model.predict_proba(image)
        print('result:', result_probability)

        # 给出类别预测(改）
        print(max(result_probability[0]))
        if max(result_probability[0]) >= 0.9:
            result = self.model.predict_classes(image)
            print('result:', result)
            # 返回类别预测结果
            return result[0]
        else:
            return -1


if __name__ == '__main__':
    classes = json_labels_read_from_file('dataset/labels.json')
    nb_classes = len(classes)
    print(nb_classes)

    train_dataset = load_dataset('dataset/train.tfrecords')
    test_dataset = load_dataset('dataset/test.tfrecords')

    model = Model()
    model.build_model(nb_classes=nb_classes)

    print(train_dataset)

    for image, label in train_dataset:
        print(image.shape)

    # 测试训练函数的代码
    model.train(train_dataset, test_dataset, 10, 50, True)

    model.save_model(file_path='./model/face.h5')
