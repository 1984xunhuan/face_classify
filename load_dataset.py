import codecs
import json
import cv2
import tensorflow as tf


IMAGE_SIZE = 64


feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'data': tf.io.FixedLenFeature([], tf.string)
}


def json_labels_read_from_file(file_path):
    with codecs.open(file_path, "r", "utf-8") as fp:
        load_dict = json.load(fp)
        print("读取出的数据为:{}".format(load_dict))
        return load_dict


def read_and_decode(example_string):
    '''
    从TFrecord格式文件中读取数据
    '''
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    image = feature_dict['data']
    label = feature_dict['label']

    image = tf.io.decode_raw(image, tf.uint8)
    print(image.shape)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    print(image.shape)

    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5 # 归一化
    label = tf.cast(label, dtype='int32')
    return image, label


def load_dataset(file_name):
    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(read_and_decode)  # 解析数据
    return dataset



# 将输入的图像大小统一
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = 0, 0, 0, 0
    # 获取图像大小
    h, w, _ = image.shape
    # 对于长宽不一的，取最大值
    longest_edge = max(h, w)
    # 计算较短的边需要加多少像素
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    # 定义填充颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant_image, (height, width))


if __name__ == '__main__':
    print(tf.__version__)
    classes = json_labels_read_from_file('dataset/labels.json')
    print(classes.get('0'))
    dataset = load_dataset('dataset/train.tfrecords')

    #for image, label in dataset:
        #print(image.numpy(), label.numpy())
        #print(label.numpy())
