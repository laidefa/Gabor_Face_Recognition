# -*- coding: utf-8 -*-
from __future__ import division
import time
time1 = time.time()
import cv2
from pylab import*
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from PIL import Image
from math import sqrt


class LBP:
    def __init__(self):
        #revolve_map为旋转不变模式的36种特征值从小到大进行序列化编号得到的字典
        self.revolve_map={0:0,1:1,3:2,5:3,7:4,9:5,11:6,13:7,15:8,17:9,19:10,21:11,23:12,
                          25:13,27:14,29:15,31:16,37:17,39:18,43:19,45:20,47:21,51:22,53:23,55:24,
                          59:25,61:26,63:27,85:28,87:29,91:30,95:31,111:32,119:33,127:34,255:35}
        #uniform_map为等价模式的58种特征值从小到大进行序列化编号得到的字典
        self.uniform_map={0:0,1:1,2:2,3:3,4:4,6:5,7:6,8:7,12:8,
                          14:9,15:10,16:11,24:12,28:13,30:14,31:15,32:16,
                          48:17,56:18,60:19,62:20,63:21,64:22,96:23,112:24,
                          120:25,124:26,126:27,127:28,128:29,129:30,131:31,135:32,
                          143:33,159:34,191:35,192:36,193:37,195:38,199:39,207:40,
                          223:41,224:42,225:43,227:44,231:45,239:46,240:47,241:48,
                          243:49,247:50,248:51,249:52,251:53,252:54,253:55,254:56,
                          255:57}


     #将图像载入，并转化为灰度图，获取图像灰度图的像素信息
    def describe(self,image):
        image_array=np.array(Image.open(image).convert('L'))
        return image_array

    #图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    #比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_basic_lbp(self,image_array,i,j):
        sum=[]
        if image_array[i-1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i-1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i+1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum

    #获取二进制序列进行不断环形旋转得到新的二进制序列的最小十进制值
    def get_min_for_revolve(self,arr):
        values=[]
        circle=arr
        circle.extend(arr)
        for i in range(0,8):
            j=0
            sum=0
            bit_num=0
            while j<8:
                sum+=circle[i+j]<<bit_num
                bit_num+=1
                j+=1
            values.append(sum)
        return min(values)

    #获取值r的二进制中1的位数
    def calc_sum(self,r):
        num=0
        while(r):
            r&=(r-1)
            num+=1
        return num

    #获取图像的LBP原始模式特征
    def lbp_basic(self,image_array):
        basic_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                bit_num=0
                result=0
                for s in sum:
                    result+=s<<bit_num
                    bit_num+=1
                basic_array[i,j]=result
        return basic_array

   #获取图像的LBP旋转不变模式特征
    def lbp_revolve(self,image_array):
        revolve_array=np.zeros(image_array.shape, np.uint8)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                sum=self.calute_basic_lbp(image_array,i,j)
                revolve_key=self.get_min_for_revolve(sum)
                revolve_array[i,j]=self.revolve_map[revolve_key]
        return revolve_array

  #获取图像的LBP等价模式特征
    def lbp_uniform(self,image_array):
        uniform_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_array[i,j]=self.uniform_map[basic_array[i,j]]
                 else:
                     uniform_array[i,j]=58
        return uniform_array

    #获取图像的LBP旋转不变等价模式特征
    def lbp_revolve_uniform(self,image_array):
        uniform_revolve_array=np.zeros(image_array.shape, np.uint8)
        basic_array=self.lbp_basic(image_array)
        width=image_array.shape[0]
        height=image_array.shape[1]
        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_revolve_array[i,j]=self.calc_sum(basic_array[i,j])
                 else:
                     uniform_revolve_array[i,j]=9
        return uniform_revolve_array


##直方图归一化
def MaxMinNormalization(hist1):
    x1=[]
    sum1=np.sum(hist1)
    for i in range(0,len(hist1)):
        x =hist1[i]/sum1
        x1.append(round(x,4))
    return x1



##基本LBP直方图
def lbp_basic_hist(gray1):
    lbp = LBP()
    basic_array1=lbp.lbp_basic(gray1)
    k1=[]
    hist11=[]
    hist1 = cv2.calcHist([basic_array1], [0], None, [256], [0, 256])
    for each in hist1:
        k1.append(int(each))
    for each in k1:
        hist11.append(int(each))
    hist11=MaxMinNormalization(hist11)
    return hist11


##等价模式直方图
def lbp_uniform_hist(gray1):
    lbp = LBP()
    basic_array1 = lbp.lbp_uniform(gray1)
    k1 = []
    hist11 = []
    hist1 = cv2.calcHist([basic_array1], [0], None, [59], [0,59])
    for each in hist1:
        k1.append(int(each))
    for each in k1:
        hist11.append(int(each))
    hist11 = MaxMinNormalization(hist11)
    return hist11




######获取基本lbp全部直方图联级特征向量
def get_imageLBP_histall(img1):
    q1=[]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    m,n=gray1.shape
    img_lbp_hist1=[]
    stepx=m/8
    srepy=n/8
    for i in range(0,m,stepx):
        for j in range(0,n,srepy):
            roi=gray1[i:i+stepx,j:j+srepy]
            m1=lbp_basic_hist(roi)
            img_lbp_hist1.append(m1)
    for each in img_lbp_hist1:
        for each1 in each:
            q1.append(each1)
    return q1



#####获取等价模式全部直方图联级特征向量
def get_imageLBP_unoform_histall(img1):
    q1=[]
    img0 = cv2.resize(img1, (96, 112), interpolation=cv2.INTER_CUBIC)
    m=112
    n=96
    img_lbp_hist1=[]
    stepx=14
    srepy=12
    for i in range(0,m,stepx):
        for j in range(0,n,srepy):
            roi=img0[i:i+stepx,j:j+srepy]
            m1=lbp_uniform_hist(roi)
            img_lbp_hist1.append(m1)
    for each in img_lbp_hist1:
        for each1 in each:
            q1.append(each1)
    return q1



######余弦距离相似度计算###########################
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return  abs((part_up / part_down))


#################person相关系数相似度计算#######################

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den

  #############################################################################




############################PCA训练图像数据##########################################

############gamma校正查找表
def BuildTable(gamma):
    table=[]
    for i in range(0,256):
        x1=(i+0.5)/256
        x2=1/gamma
        x3=np.power(x1,x2)
        x4=x3*256-0.5
        table.append(x4)
    return table


#############图像预处理：gamma校正算法
def GammaCorrectiom(img1,gamma):
    mm=BuildTable(gamma)
    m, n = img1.shape
    for i in range(0, m):
        for j in range(0, n):
            img1[i][j] = mm[img1[i][j]]
    return img1



#############DoG滤波##################
def DoG(img1,sig1,sig2):
    img2= cv2.GaussianBlur(img1, (3, 3),sig1) - cv2.GaussianBlur(img1, (3, 3), sig2)
    return img2


##图片预处理,归一化,gamma校正,DoG滤波,均衡化
def image_processing(img1):
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ##归一化
    img0 = cv2.resize(img1, (96, 112), interpolation=cv2.INTER_CUBIC)
    # ##gamma校正
    # img2=GammaCorrectiom(img0,0.8)
    # ##DoG滤波
    # img3=DoG(img0,0.4,0.3)
    #直方图均衡化
    roi_gray1= cv2.equalizeHist(img0)
    return roi_gray1




def process_img(image1):
    roi_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(roi_gray1, (96, 112), interpolation=cv2.INTER_CUBIC)
    # img2 = GammaCorrectiom(img1, 0.45)
    # img3 = cv2.GaussianBlur(img2, (3, 3), 0.4) - cv2.GaussianBlur(img2, (3, 3), 0.3)
    img4 = cv2.equalizeHist(img1)
    return img4






# define PCA
def pca(data,k):
    data = float32(mat(data))
    rows,cols = data.shape
    data_mean = mean(data,0)#对列求均值
    data_mean_all = tile(data_mean,(rows,1))
    Z = data - data_mean_all
    T1 = Z*Z.T #使用矩阵计算，所以前面mat
    D,V = linalg.eig(T1) #特征值与特征向量
    V1 = V[:,0:k]#取前k个特征向量
    V1 = Z.T*V1
    for i in xrange(k): #特征向量归一化
        L = linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L

    data_new = Z*V1 # 降维后的数据
    return data_new,data_mean,V1



def img2vector1(filename1):
    img1=cv2.imread(filename1)
    img=process_img(img1)
    imgVector =get_imageLBP_unoform_histall(img)
    return imgVector

def img2vector2(filename):
    img = cv2.imread(filename)
    img1=image_processing(img)
    imgVector =get_imageLBP_unoform_histall(img1)
    return imgVector

def loadDataSet(m,k):
    dataSetDir = 'E:/att_faces'
    choose = random.permutation(10) + 1
    train_face = zeros((m * k,8*8*59))
    for i in xrange(m):
        print i
        people_num = i + 1
        for j in xrange(10):
            if j < k:
                print "第"+str(j)+"个"
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(choose[j]) + '.pgm'
                img = img2vector2(filename)
                train_face[i*k+j, :] = img

    return train_face,people_num


def facefind():
    train_face, people_num=loadDataSet(40,9)
    data_train_new, data_mean, V = pca(train_face,30)
    return data_mean, V


def hist_get(image1,image2):
    img1 = process_img(image1)
    img2 = process_img(image2)
    imgVector1 = get_imageLBP_unoform_histall(img1)
    print imgVector1
    imgVector2 = get_imageLBP_unoform_histall(img2)
    print imgVector2
    data_mean, V1 = facefind()
    imgVector11 = imgVector1 - tile(data_mean, (1, 1))
    p1 = imgVector11 * V1
    p1 = p1.getA()
    for each in p1:
        p11 = each
        print p11
    imgVector22 = imgVector2 - tile(data_mean, (1, 1))
    p2 = imgVector22 * V1
    p2 = p2.getA()
    for each in p2:
        p22 = each
        print p22
    print "Image similarity is:%s" % cos_dist(p11, p22)
    if cos_dist(p11,p22)>0.6:
        print "是同一人"
    else:
        print "不是同一人"
    # print "Image similarity is:%s" % corrcoef(p11, p22)
    cv2.imshow("basic_array1", img1)
    cv2.imshow("basic_array2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_detection_OCR(pic_path):
    faceCascade = cv2.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray= cv2.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(5,20))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y +h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_CUBIC)
    return roi_gray



def image_detection_identity(pic_path):
    faceCascade = cv2.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray = cv2.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(1,100))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + 112, x+10:x + 102]
    return roi_gray







if __name__ == '__main__':
    #
    filename1 = "D:/yicun/7.jpg"
    filename2 = "D:/yicun/8.jpg"
    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)




    hist_get(image1, image2)

    time2 = time.time()
    print u'ok,程序结束!'
    print u'总共耗时：' + str(time2 - time1) + 's'

    # # ##############################头像###################
    # # #
    # image1=image_detection_identity("E:/1013.png")
    # img1 = cv2.resize(image1, (96, 112), interpolation=cv2.INTER_CUBIC)
    # # # image2 = image_detection_OCR("E:/10183.jpg")
    # # # img2 = cv2.resize(image2, (96, 112), interpolation=cv2.INTER_CUBIC)
    # # # hist_get(img1, img2)
    # #
    # # # #
    # #
    # #
    # # # #######################身份证#######################
    # # image1 = image_detection_identity("E:/770.jpg")
    # # img1 = cv2.resize(image1, (96, 112), interpolation=cv2.INTER_CUBIC)
    # image2 = image_detection_identity("E:/3106.png")
    # img2 = cv2.resize(image2, (96, 112), interpolation=cv2.INTER_CUBIC)
    # # #
    # hist_get(img1, img2)

    cv2.imshow("img1", image1)
    cv2.imshow("img2",image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

