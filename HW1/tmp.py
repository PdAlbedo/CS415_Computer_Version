import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


imgPath_2 = "art.png"
img_2 = plt.imread(imgPath_2)
print(img_2)

def Median_filtering(image,window_size):   #image为传入灰度图像，window_size为滤波窗口大小
    high, wide = image.shape
    img = image.copy()
    mid = (window_size-1) // 2
    med_arry = []
    for i in range(high-window_size):
        for j in range(wide-window_size):
            for m1 in range(window_size):
                for m2 in range(window_size):
                    med_arry.append(image[i+m1,j+m2])

            # for n in range(len(med_arry)-1,-1,-1):
            med_arry.sort()   				#对窗口像素点排序
            # print(med_arry)
            img[i+mid,j+mid] = med_arry[(len(med_arry)+1) // 2]        #将滤波窗口的中值赋给滤波窗口中间的像素点
            del med_arry[:]

    return img

imgPath_2 = "art.png"
img_2 = plt.imread(imgPath_2)
print(img_2)
dst = Median_filtering(img_2, 5)
plt.imshow(dst)
plt.show()
