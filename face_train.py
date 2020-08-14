########################
# 人脸特征训练
########################

import os
import random
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from load_dataset import load_dataset, resize_image, IMAGE_SIZE, json_labels_read_from_file

import cv2
from PIL import Image

'''
对数据集的处理，包括：
1、加载数据集
2、将数据集分为训练集、验证集和测试集
3、根据Keras后端张量操作引擎的不同调整数据维度顺序
4、对数据集中的标签进行One-hot编码
5、数据归一化
'''
class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序，包括rows，cols，channels，用于后续卷积神经网络模型中第一层卷积层的input_shape参数
        self.input_shape = None

        # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # 加载数据集到内存
        dataset = load_dataset(self.path_name)

        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 100))

        # tensorflow 作为后端，数据格式约定是channel_last，与这里数据本身的格式相符，如果是channel_first，就要对数据维度顺序进行一下调整
        if K.image_data_format == 'channel_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
        # 输出训练集、验证集和测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')
        # 后面模型中会使用categorical_crossentropy作为损失函数，这里要对类别标签进行One-hot编码
        train_labels = keras.utils.to_categorical(train_labels, nb_classes)
        test_labels = keras.utils.to_categorical(test_labels, nb_classes)

        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels

# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, nb_classes=2):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))  # 当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含batch_size）
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # strides默认等于pool_size
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # 输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size=64, nb_epoch=15, data_augmentation=True):
        logdir='./logs'
        checkpoint_path='./checkpoint/face.{epoch:02d}-{val_loss:.2f}.ckpt'

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                           histogram_freq=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               save_weights_only=True,
                                               verbose=1,
                                               period=5)
        ]


        self.model.compile(loss='categorical_crossentropy',
                           optimizer='ADAM',
                           metrics=['accuracy'])

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_split=0.2,
                           callbacks=callbacks,
                           shuffle=True)
        # 使用实时数据提升
        else:
            datagen = ImageDataGenerator(rotation_range=20,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         horizontal_flip=True)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
                                     epochs=nb_epoch,
                                     callbacks=callbacks,
                                     validation_data=(dataset.test_images, dataset.test_labels),
                                     shuffle=True)

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
        #image = image.astype('float32')
        #image /= 255
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 归一化

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

    dataset = Dataset('dataset/train.tfrecords')
    dataset.load(nb_classes=nb_classes)

    model = Model()
    model.build_model(nb_classes=nb_classes)

    print(dataset)

    # 测试训练函数的代码
    model.train(dataset, 64, 15, False)
    model.evaluate(dataset)
    model.save_model(file_path='./model/face.h5')
