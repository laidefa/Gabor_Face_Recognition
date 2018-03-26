# -*- coding: utf-8 -*-
from __future__ import division
import time
time1 = time.time()
import cv2 as cv
from pylab import *
from numpy import *
import numpy as np


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


#############gamma校正算法
def GammaCorrectiom(img1,gamma):
    mm=BuildTable(gamma)
    m, n = img1.shape
    for i in range(0, m):
        for j in range(0, n):
            img1[i][j] = mm[img1[i][j]]
    return img1


#############DoG滤波##################
def DoG(img1,sig1,sig2):
    img2= cv.GaussianBlur(img1, (3, 3),sig1) - cv.GaussianBlur(img1, (3, 3), sig2)
    return img2



#定义了一个5尺度8方向的Gabor变换
def img2vector(image):
    hist1=[]
    #图像预处理
    image=cv.imread(image,1)
    image = cv.resize(image, (92, 112), interpolation=cv.INTER_CUBIC)
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    src=GammaCorrectiom(src,0.8)
    src=DoG(src,0.9,0.3)
    src=cv.equalizeHist(src)
    ##gabor变换
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.
    us=[7,12,17,21,26]             #5种尺度
    # vs=[0,30,60,90,120,150]     #6个方向
    vs=[0,pi/4,2*pi/4, 3*pi/4, 4*pi/4, 5*pi/4, 6*pi/4,7*pi/4]#8个方向
    kernel_size =21
    sig = 5                     #sigma 带宽，取常数5
    gm = 1.0                    #gamma 空间纵横比，一般取1
    ps = 0.0                    #psi 相位，一般取0
    i=0
    for u in us:
        for v in vs:
            lm = u
            th = v*np.pi/180
            kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
            kernelimg = kernel/2.+0.5
            dest = cv.filter2D(src_f, cv.CV_32F,kernel)
            dst=np.power(dest,2)
            p1 = dst[20:50, 10:82]
            p2 = dst[50:100, 21:71]
            for each in p1:
                for each1 in each:
                    hist1.append(each1)
            for each0 in p2:
                for each2 in each0:
                    hist1.append(each2)

    return hist1




# define PCA
def pca(data,k):
    data = float32(mat(data))
    rows,cols = data.shape#取大小
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




#load dataSet
def loadDataSet(k):  #choose k(0-10) people as traintest for everyone
    ##step 1:Getting data set
    print "--Getting data set---"
    #note to use '/'  not '\'
    dataSetDir = 'E:/att_faces'
    #显示文件夹内容
    choose = random.permutation(10)+1 #随机排序1-10 (0-9）+1
    # print choose
    train_face = zeros((40*k,4660*40))
    train_face_number = zeros(40*k)
    test_face = zeros((40*(10-k),4660*40))
    test_face_number = zeros(40*(10-k))
    for i in xrange(40): #40 sample people
        print i
        people_num = i+1
        for j in xrange(10): #everyone has 10 different face
            if j < k:
                filename = dataSetDir+'/s'+str(people_num)+'/'+str(choose[j])+'.pgm'
                img = img2vector(filename)
                train_face[i*k+j,:] = img
                train_face_number[i*k+j] = people_num
            else:
                filename = dataSetDir+'/s'+str(people_num)+'/'+str(choose[j])+'.pgm'
                img = img2vector(filename)
                test_face[i*(10-k)+(j-k),:] = img
                test_face_number[i*(10-k)+(j-k)] = people_num

    return train_face,train_face_number,test_face,test_face_number


# calculate facefind
def facefind():
    # Getting data set
    # 选择每个人中随机9张作为训练集
    train_face,train_face_number,test_face,test_face_number = loadDataSet(9)
    # 选择每个人中随机9张作为训练集，其他的作为测试集，并且降维到30维
    data_train_new,data_mean,V = pca(train_face,100)
    return  data_mean,V


######计算相似度
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



#######两张人脸图片对比
def get_hist(filename1,filename2):
    img11=cv.imread(filename1)
    img22=cv.imread(filename2)
    img11=cv.resize(img11, (92,112), interpolation=cv.INTER_CUBIC)
    img22=cv.resize(img22, (92,112), interpolation=cv.INTER_CUBIC)
    img11=cv.cvtColor(img11, cv.COLOR_BGR2GRAY)
    img22=cv.cvtColor(img22, cv.COLOR_BGR2GRAY)
    imgVector1=img2vector(filename1)
    imgVector2=img2vector(filename2)
    data_mean,V1=facefind()
    imgVector11 = imgVector1 - tile(data_mean, (1, 1))
    p1 = imgVector11 * V1
    p1 = p1.getA()
    for each in p1:
        p11 = each
    imgVector22 = imgVector2 - tile(data_mean, (1, 1))
    p2 = imgVector22 * V1
    p2 = p2.getA()
    for each in p2:
        p22 = each
    print "Image similarity is:%s" % cos_dist(p11, p22)
    if cos_dist(p11, p22)>0.6:
        print "很相似，是同一个人！"
    else:
        print "不太像，不是同一个人！"
    cv.imshow("img11",img11)
    cv.imshow("img22",img22)
    cv.waitKey(0)
    cv.destroyAllWindows()



def image_detection_OCR(pic_path):
    faceCascade = cv.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray= cv.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(5,20))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y +h, x:x+w]
        roi_gray=cv.resize(roi_gray, (92, 112), interpolation=cv.INTER_CUBIC)

    return roi_gray


def image_detection_identity(pic_path):
    faceCascade = cv.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray = cv.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(1,100))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + 112, x+10:x + 102]
    return roi_gray


def get_hist2(filename1,filename2):




    img11=image_detection_identity(filename1)
    img22 = cv.imread(filename2)

    img22=image_detection_OCR(filename2)

    img11=cv.resize(img11, (92,112), interpolation=cv.INTER_CUBIC)
    img22=cv.resize(img22, (92,112), interpolation=cv.INTER_CUBIC)
    img11=cv.cvtColor(img11, cv.COLOR_BGR2GRAY)
    img22=cv.cvtColor(img22, cv.COLOR_BGR2GRAY)
    imgVector1=img2vector(filename1)
    imgVector2=img2vector(filename2)
    data_mean,V1=facefind()
    imgVector11 = imgVector1 - tile(data_mean, (1, 1))
    p1 = imgVector11 * V1
    p1 = p1.getA()
    for each in p1:
        p11 = each
    imgVector22 = imgVector2 - tile(data_mean, (1, 1))
    p2 = imgVector22 * V1
    p2 = p2.getA()
    for each in p2:
        p22 = each
    print "Image similarity is:%s" % cos_dist(p11, p22)
    if cos_dist(p11, p22)>0.6:
        print "很相似，是同一个人！"
    else:
        print "不太像，不是同一个人！"
    cv.imshow("img11",img11)
    cv.imshow("img22",img22)
    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == '__main__':

    filename1="E:/yale/s5.bmp"
    filename2="E:/yale/s6.bmp"
    # img1=image_detection_identity(filename1)
    # img2= image_detection_OCR(filename2)
    # get_hist2(filename1, filename2)
    # img2=image_detection_identity(filename2)

    get_hist(filename1, filename2)

    # cv.imshow("img1",img1)
    # cv.imshow("img2", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    time2 = time.time()
    print u'ok,程序结束!'
    print u'总共耗时：' + str(time2 - time1) + 's'

