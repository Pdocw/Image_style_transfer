import cv2
import math
import numpy as np
import face_alignment
from matplotlib import pyplot as plt
from removebg import RemoveBg
class FaceDetect:#人脸检测
    def __init__(self, device, detector):

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)  #保存了脸部信息的特征点的坐标
        #print(landmarks)  
        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def crop(self, image, landmarks):
        return self.__crop(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)#保存了脸部信息的特征点的坐标
        #print(preds)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # 如果有很多张脸就找到最大的那张
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])#纵坐标的最小值
                landmarks_bottom = np.max(pred[:, 1])#纵坐标的最大值
                landmarks_left = np.min(pred[:, 0])#横坐标的最小值
                landmarks_right = np.max(pred[:, 0])#横坐标的最大值
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)#选取面积最大的脸
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # 旋转角度 根据两只眼睛调整角度
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # 旋转后的图像尺寸
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # 映射矩阵
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])  # 2*3

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
        # image 输入图像
        # M  变换矩阵
        # (new_w, new_h) 输出图像的大小
        # borderValue 边界填充值

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)#每个点后面加入一个1  3*n
        print(landmarks)
        landmarks_rotate = np.dot(M, landmarks.T).T #映射之后的点
        print(landmarks_rotate) #映射之后的点
        return image_rotate, landmarks_rotate


    @staticmethod
    def __crop(image, landmarks):#裁剪出头像附近的区域
        # 特征点的区域范围
        landmarks_top = np.min(landmarks[:, 1]) #纵坐标的最小值
        landmarks_bottom = np.max(landmarks[:, 1]) #纵坐标的最大值
        landmarks_left = np.min(landmarks[:, 0]) #横坐标的最小值
        landmarks_right = np.max(landmarks[:, 0]) #横坐标的最大值

        # 扩大脸的范围  
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))  # +0.8
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top)) # -0.3
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left)) # -0.3 
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left)) # -0.3 

        if bottom - top > right - left:  #如果长大于宽
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:   #如果长小于宽
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255  #建立一个全白的矩阵图

        h, w = image.shape[:2]

        #防止越界
        left_white = max(0, -left)  # 最终边界的最左边
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)


        top_white = max(0, -top) # 最终边界的最上方
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop

def transparence2white(img):#将带透明度的png图转为背景为白色的jpg图
    width=img.shape[0]  # 宽度
    height=img.shape[1]  # 高度
    for y in range(height):
        for x in range(width):
            # 遍历图像每一个点，获取到每个点4通道的颜色数据
            if(img[x,y][3]==0):  # 最后一个通道为透明度，如果其值为0，即图像是透明
                img[x,y]=[255,255,255,255]  # 则将当前点的颜色设置为白色，且图像设置为不透明
    return img

if __name__ == '__main__':
    # rmbg = RemoveBg("DgxdCVqHTGSiVYuRTxriecY3***", "error.log") # 引号内是获取的API
    # rmbg.remove_background_from_img_file("zxy.png") #图片地址
    img=cv2.imread('zxy.png_no_bg.png',-1)  # 读取图片。-1将图片透明度传入，数据由RGB的3通道变成4通道
    img=transparence2white(img)  # 将图片传入，改变背景色后，返回
    cv2.imwrite('zxy.png_no_bg.png',img)  # 保存图片，文件名自定义，也可以覆盖原文件
    img = cv2.cvtColor(cv2.imread('zxy.png_no_bg.png'), cv2.COLOR_BGR2RGB)
    fd = FaceDetect(device='cpu',detector='sfd')
    face_info = fd.align(img)
    
    #face_info[0]保存了旋转之后的图片
    #face_info[1]中保存了脸部信息的点
    face = fd.crop(face_info[0],face_info[1])
    plt.imshow(face)
    plt.show()
    cv2.imwrite('zxy_face.png', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    if face_info is not None:
        image_align, landmarks_align = face_info

        for i in range(landmarks_align.shape[0]):#将人脸特征点用红色点标记出来
            cv2.circle(image_align, (int(landmarks_align[i][0]), int(landmarks_align[i][1])), 2, (255, 0, 0), -1)

        cv2.imwrite('zxy_align.png', cv2.cvtColor(image_align, cv2.COLOR_RGB2BGR))
