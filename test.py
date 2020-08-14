import sys
from face_train import Model
from load_dataset import load_dataset, resize_image, IMAGE_SIZE, json_labels_read_from_file
from PIL import Image
import tensorflow as tf

if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='./model/face.h5')

    classes = json_labels_read_from_file('dataset/labels.json')

    img_path = './data/temp/1.jpg'
    img = Image.open(img_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # resize图片大小
    image = img.tobytes()  # 将图片转化为原生bytes
    image = tf.io.decode_raw(image, tf.uint8)
    print(image.shape)
    image = tf.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    print(image.shape)
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 归一化

    faceID = model.face_predict(image)

    print('faceID: %d' % faceID)
    print(classes.get(str(faceID)))
