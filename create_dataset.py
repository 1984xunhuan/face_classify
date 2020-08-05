import os
import tensorflow as tf
from PIL import Image
import json
import codecs

# 源数据地址
cwd = 'data'
# 生成record路径及文件名
train_record_path = "dataset/train.tfrecords"
test_record_path = "dataset/test.tfrecords"
# 分类
#classes = {'0': 'dingcuixiao', '1': 'dinggaotao', '2': 'dingshaolan', '3': 'lvxiaoping', '4': 'zhangqing'}


def json_labels_read_from_file(file_path):
    with codecs.open(file_path, "r", "utf-8") as fp:
        load_dict = json.load(fp)
        print("读取出的数据为:{}".format(load_dict))
        return load_dict


classes = json_labels_read_from_file('dataset/labels.json')

IMAGE_SIZE = 64


def create_train_record():
    """创建训练集tfrecord"""
    writer = tf.io.TFRecordWriter(train_record_path)  # 创建一个writer
    NUM = 1  # 显示创建过程（计数）

    for index, name in classes.items():
        index = int(index)
        print("index: %d, name: %s" % (index, name))
        class_path = cwd + "/" + name + '/'
        l = int(len(os.listdir(class_path)) * 0.7)  # 取前70%创建训练集
        print('==============l: %d' % l)
        for img_name in os.listdir(class_path)[:l]:
            img_path = class_path + img_name
            print(img_path)

            img = Image.open(img_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # resize图片大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes

            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(img_raw)]))
                    }
                )
            )
            writer.write(exam.SerializeToString())
            print('Creating train record in ', NUM)
            NUM += 1
    writer.close()  # 关闭writer
    print("Create train_record successful!")


def create_test_record():
    """创建测试tfrecord"""
    writer = tf.io.TFRecordWriter(test_record_path)
    NUM = 1
    for index, name in classes.items():
        index = int(index)
        print("index: %d, name: %s" % (index, name))
        class_path = cwd + '/' + name + '/'
        l = int(len(os.listdir(class_path)) * 0.7)
        for img_name in os.listdir(class_path)[l:]:  # 剩余30%作为测试集
            img_path = class_path + img_name

            img = Image.open(img_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # resize图片大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes

            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(img_raw)]))
                    }
                )
            )
            writer.write(exam.SerializeToString())
            print('Creating test record in ', NUM)
            NUM += 1
    writer.close()
    print("Create test_record successful!")


def read_record(filename):
    """读取tfrecord"""
    filename_queue = tf.train.string_input_producer([filename])  # 创建文件队列
    reader = tf.TFRecordReader()  # 创建reader
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['data']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    img = tf.cast(img, tf.float32) * (1. / 255)  # 归一化
    label = tf.cast(label, tf.int32)
    return img, label


def get_batch_record(filename, batch_size):
    """获取batch"""
    image, label = read_record(filename)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],  # 随机抽取batch size个image、label
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=1000)
    return image_batch, label_batch


def json_labels_write_to_file(file_path, dict_data):
    with open(file_path, mode='w', encoding='utf-8') as fp:
        json.dump(dict_data, fp, ensure_ascii=False, indent=2)


def main():
    create_train_record()
    create_test_record()


if __name__ == '__main__':
    main()

    #json_labels_write_to_file('dataset/labels.json', classes)
    #classes = json_labels_read_from_file('dataset/labels.json')
    #print(classes)

