"""
Xichen Liu
CS415 HW5
Image classification
"""

import os
import sys
import cv2
import numpy as np

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)

class Descriptor:
    def __init__(self, vector = None, label = None):
        self.vector = vector
        self.label = label

class Image:
    def __init__(self, img = None, name = None, key_p = None, des = None, hist = None):
        self.image = img
        self.image_name = name
        self.key_point = key_p
        self.descriptor = des
        self.label = None
        self.histogram = hist

class KnnClassifier:
    def __init__(self, k = 1):
        self.k = k

    def predict(self, train_imgs, test_img):




def update_images(imgs):
    key_des = []
    sift = cv2.SIFT_create()
    for i in range(len(imgs)):
        img_gray = cv2.cvtColor(imgs[i].image, cv2.COLOR_BGR2GRAY)
        k_and_d = (sift.detectAndCompute(img_gray, None))
        imgs[i].key_point = k_and_d[0]
        ttl = []
        for j in k_and_d[1]:
            des_ele = Descriptor(j, None)
            ttl.append(des_ele)
        imgs[i].descriptor = ttl
        key_des.append(k_and_d)

    return imgs, key_des

def coalesce_des(key_des):
    tmp = [key_des[i][1] for i in range(0, len(key_des)) if key_des[i][1] is not None]

    ttl_len = 0
    for i in range(len(key_des)):
        ttl_len += len(key_des[i][0])

    des_num = len(tmp[0][0])
    des = np.zeros((ttl_len, des_num))

    c = 0
    for i in tmp:
        for j in i:
            des[c] = j
            c += 1

    return des

def k_means(des, words):
    # K-means algorithm parameter
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    com, lab, center = cv2.kmeans(des, words, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)

    return com, lab, center

def set_labels(imgs, lab, des_num, word_bag):

    idx = 0
    for i in imgs:
        i.histogram = np.zeros(des_num, dtype = object)
        if i.descriptor is not None:
            for j in i.descriptor:
                j.label = lab[idx][0]
                i.histogram[lab[idx][0]] += 1
                idx += 1
        else:
            i.histogram = np.zeros(word_bag, dtype = np.float32)

    return imgs


if __name__ == '__main__':

    train_images = []
    # for file_name in os.listdir('.\\data\\train\\TallBuilding'):
    #     img = cv2.imread('.\\data\\train\\TallBuilding\\' + file_name)
    for dir_name in os.listdir('.\\data\\train'):
        for file_name in os.listdir('.\\data\\train\\' + dir_name):
            img = cv2.imread('.\\data\\train\\' + dir_name + '\\' + file_name)

            train_images.append(Image(img, os.path.splitext(file_name)[0]))

    train_images, train_images_key_des = update_images(train_images)

    descriptor = coalesce_des(train_images_key_des).astype('float32')

    words_num = 150
    compact, label, centroids = k_means(descriptor, words_num)

    train_images = set_labels(train_images, label, words_num, words_num)

    print(train_images[1].histogram)
