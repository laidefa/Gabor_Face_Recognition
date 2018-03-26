# -*- coding: utf-8 -*-
from __future__ import division
import time
time1 = time.time()
import cv2
import numpy as np
from numpy import *

#######照片人脸检测
def image_detection_OCR(pic_path):
    faceCascade = cv2.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray= cv2.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(5,20))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        # print x,y,w,h
        roi_gray = gray[y-15:y+h+4, x:x+w+4]
        roi_gray=cv2.resize(roi_gray, (112, 128), interpolation=cv2.INTER_CUBIC)
    return roi_gray


#######身份证人脸检测
def image_detection_identity(pic_path):
    faceCascade = cv2.CascadeClassifier("E:/haarcascade_frontalface_default.xml")
    gray = cv2.imread(pic_path)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(1,100))
    print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        # print x,y,w,h
        roi_gray = gray[y-20:y +108, x-2:x + 110]
    return roi_gray



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


######图像预处理
def image_process(img1):
    print img1.shape
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_gray1 = GammaCorrectiom(img1_gray,0.8)
    return img1_gray1



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

def showImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def hog_detector(img):
    height, width = img.shape
    gradient_magnitude, gradient_angle = calc_gradient(img)
    #showImage(gradient_magnitude)
    hog_vector = []
    num_horizontal_blocks = int(width/8)
    num_vertical_blocks = int(height/8)
    cell_gradients = np.zeros([int(height/8), int(width/8)])
    cell_gradient_vector = []
    #  calculating cell gradient vector
    for i in range(0, (height - (height%8)-1), 8):
        horizontal_vector = []
        for j in range(0, width - (width%8) -1, 8):
            cell_magnitude = gradient_magnitude[i:i+8, j:j+8]
            cell_angle = gradient_angle[i:i+8, j:j+8]
            horizontal_vector.append(calc_cell_gradient(cell_magnitude, cell_angle))
        cell_gradient_vector.append(horizontal_vector)

    # print "rendering height, width", height, width
    render_gradient(np.zeros([height, width]), cell_gradient_vector)
    height = len(cell_gradient_vector)
    width = len(cell_gradient_vector[0])
    #  calculating final gradient hog vector
    for i in range(height-1):
        for j in range(width - 1):
            vector = []
            vector.extend(cell_gradient_vector[i][j])
            vector.extend(cell_gradient_vector[i][j+1])
            vector.extend(cell_gradient_vector[i+1][j])
            vector.extend(cell_gradient_vector[i+1][j+1])
            mag = lambda vector: math.sqrt(sum(i**2 for i in vector))
            magnitude = mag(vector)
            if magnitude != 0:
                normalize = lambda vector, magnitude : [element/magnitude for element in vector]
                vector = normalize(vector, magnitude)

            hog_vector.append(vector)


    return hog_vector



# calculate gradient magnitude and gradient angle for image using sobel
def calc_gradient(img):
    gradient_values_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    cv2.convertScaleAbs(gradient_values_x)
    gradient_values_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    cv2.convertScaleAbs(gradient_values_y)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    return gradient_magnitude, gradient_angle


#   calculate gradient for cells into a vector of 9 values
def calc_cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = dict([(20, 0), (60, 0),  (100, 0),  (140, 0), (180, 0),  (220, 0), (260, 0) , (300, 0), (340, 0)])
    x_left = 0
    y_left = 0
    cell_magnitude = abs(cell_magnitude)
    for i in range(x_left, x_left+8):
        for j in range(y_left, y_left+8):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle = get_closest_bins(gradient_angle, orientation_centers)
            #print gradient_angle, min_angle, max_angle
            if min_angle == max_angle:
                orientation_centers[min_angle] += gradient_strength
            else:
                orientation_centers[min_angle] += gradient_strength * (abs(gradient_angle - max_angle)/40)
                orientation_centers[max_angle] += gradient_strength * (abs(gradient_angle - min_angle)/40)
    cell_gradient = []
    for key in orientation_centers:
        cell_gradient.append(orientation_centers[key])
    return cell_gradient





def get_closest_bins(gradient_angle, orientation_centers):
    angles = []
    #print math.degrees(gradient_angle)
    for angle in orientation_centers:
        if abs(gradient_angle - angle) < 40:
            angles.append(angle)
    angles.sort()
    if len(angles) == 1:
        #print gradient_angle, angles[0]
        return angles[0], angles[0]

    return angles[0], angles[1]

#   render gradient
def render_gradient(image, cell_gradient):

    height, width = image.shape
    height = height - (height%8) - 1
    width = width - (width%8) - 1
    x_start = 4
    y_start = 4
    #x_start = height - 8
    #y_start = width - 8
    cell_width = 4
    for x in range(x_start, height, 8):
        for y in range(y_start, width, 8):
            cell_x =int(x/8)
            cell_y = int(y/8)
            cell_grad = cell_gradient[cell_x][cell_y]
            mag = lambda vector: math.sqrt(sum(i**2 for i in vector))
            normalize = lambda vector, magnitude : [element/magnitude for element in vector] if magnitude > 0 else  vector
            cell_grad = normalize(cell_grad, mag(cell_grad))
            angle = 0
            angle_gap = 40
            # print x, y, cell_grad
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x + magnitude*cell_width*math.cos(angle_radian))
                y1 = int(y + magnitude*cell_width*math.sin(angle_radian))
                x2 = int(x - magnitude*cell_width*math.cos(angle_radian))
                y2 = int(y - magnitude*cell_width*math.sin(angle_radian))
                test_a = (x1, y1)
                test_b = (x2, y2)
                # print  test_a
                # print test_b
                cv2.line(image, (y1, x1), (y2, x2), 255)
                angle += angle_gap
    h, w = image.shape
    # showImage(image)



if __name__ == '__main__':


    # img1=image_detection_identity("E:/1013.png")
    # img2=image_detection_OCR("E:/316.jpg")
    # img2=image_detection_OCR("E:/754.png")


    #
    img1=cv2.imread("D:/yicun/20.jpeg")
    img1 = cv2.resize(img1, (96,96), interpolation=cv2.INTER_CUBIC)
    #
    img2=cv2.imread("D:/yicun/22.jpeg")
    img2 = cv2.resize(img2, (96,96), interpolation=cv2.INTER_CUBIC)

    roi_gray1=image_process(img1)
    roi_gray2=image_process(img2)
    hist1=hog_detector(roi_gray1);
    hist2=hog_detector(roi_gray2);

    print len(hist1)
    print len(hist2)
    hist11=[]
    for each in hist1:
        for k in each:
            hist11.append(k)
    hist22=[]
    for each in hist2:
        for k in each:
            hist22.append(k)
    print "the Image similary is %s" % cos_dist(hist11,hist22)



    # cv2.imshow("identity", img1)
    # cv2.imshow("OCR", img2)

    cv2.imshow("identity", roi_gray1)
    cv2.imshow("OCR", roi_gray2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    time2 = time.time()
    print u'ok,程序结束!'
    print u'总共耗时：' + str(time2 - time1) + 's'







