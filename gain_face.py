################
# 获取人的脸部信息，并保存到所属文件夹
################


import cv2
import sys
import os


def create_folder(path):
    # 去除首位空格
    del_path_space = path.strip()
    # 去除尾部'\'
    del_path_tail = del_path_space.rstrip('\\')
    # 判读输入路径是否已存在
    isexists = os.path.exists(del_path_tail)
    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False


def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    # 检查输入路径是否存在——不存在就创建
    create_folder(path_name)

    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                if w > 200:

                    # 将当前帧保存为图片
                    img_name = '%s/%d.png' % (path_name, num)

                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    # image = grey[y:y + h, x:x + w]  # 保存灰度人脸图
                    if image.size > 0:
                        cv2.imwrite(img_name, image)

                    num += 1
                    if num >= catch_pic_num:  # 如果超过指定最大保存数量退出循环
                        break

                    # 画出矩形框的时候稍微比识别的脸大一圈
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                    # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'num:%d' % num, (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num >= catch_pic_num:
            break

        cv2.startWindowThread()  # 加在这个位置
        # 显示图像
        cv2.imshow(window_name, frame)

        # 按键盘‘q’中断采集
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    print('cap.release')
    cap.release()
    print('cv2.destroyAllWindows')
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# 判断本程序是独立运行还是被调用
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("gain face", 0, 200, './data/dgt001')
