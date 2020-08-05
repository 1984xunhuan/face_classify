# face_classify
基于opencv4.4 + tensorflow2.2.0 实现人脸分类，包括数据采集、数据集制作、训练、预测等模块。

1、user_info_collection.py 收集用户人脸图片信息
2、修改dataset/labels.json 文件分类
3、create_dataset.py  采用tfrecrod格式，制作训练和验证数据集
4、face_train.py   训练
5、face_recognition.py  预测
