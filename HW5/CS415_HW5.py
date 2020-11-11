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


class Image:
    def __init__(self, img = None, name = None, idx = 0, key_p = None,
                 des = None, lab = None, hist = None, predict = None):
        self.image = img
        self.image_name = name
        self.ID = idx
        self.key_point = key_p
        self.descriptor = des
        self.label = lab
        self.histogram = hist
        self.predicted_label = predict


class KnnClassifier:
    def __init__(self, k = 1):
        self.k = k

    def predict(self, train_imgs, test_imgs):

        for test_img in test_imgs:
            classify_dis = {}

            for train_img in train_imgs:
                dis = np.linalg.norm(test_img.histogram - train_img.histogram)
                classify_dis[str(train_img.ID)] = dis

            classify_dis = {k: v for k, v in sorted(classify_dis.items(), key = lambda item: item[1])}
            # print(classify_dis)  TODO REMOVE
            # print('_______________________________')  TODO REMOVE

            k_imgs_labels = []
            for i in range(self.k):
                nearest_idx = list(classify_dis.keys())[i]
                # print(type(np.array(nearest_img)))  TODO REMOVE
                # print('_______________________________')  TODO REMOVE
                for j in train_imgs:
                    if j.ID == int(nearest_idx):
                        # print(type(np.array(nearest_idx)))  TODO REMOVE
                        # print('_______________________________')  TODO REMOVE
                        k_imgs_labels.append(j.label)

            label_frequency = {}
            # print(len(k_imgs_labels))  TODO REMOVE
            # print('_______________________________')  TODO REMOVE
            for i in range(len(k_imgs_labels)):
                if k_imgs_labels[i] not in label_frequency.keys():
                    label_frequency[str(k_imgs_labels[i])] = 0
                else:
                    label_frequency[str(k_imgs_labels[i])] += 1

            # print(label_frequency)  TODO REMOVE
            # print('_______________________________')  TODO REMOVE
            label_frequency = {k: v for k, v in sorted(label_frequency.items(), key = lambda item: item[1])}
            # print(label_frequency)  TODO REMOVE
            # print('_______________________________')  TODO REMOVE

            label_idx = list(label_frequency.keys())[-1]
            # print(len(list(label_frequency.keys())))  TODO REMOVE
            # print('_______________________________')  TODO REMOVE
            test_img.predicted_label = label_idx

        return test_imgs


def update_images(imgs):
    key_des = []
    sift = cv2.SIFT_create()
    for i in range(len(imgs)):
        img_gray = cv2.cvtColor(imgs[i].image, cv2.COLOR_BGR2GRAY)
        k_and_d = (sift.detectAndCompute(img_gray, None))
        imgs[i].key_point = k_and_d[0]
        imgs[i].descriptor = k_and_d[1]
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    com, lab, center = cv2.kmeans(des, words, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)

    return com, lab, center


def histogram_made(imgs, center, his_len):
    for img in imgs:
        img.histogram = np.zeros(his_len)

        if img.descriptor is not None:
            for vector in img.descriptor:
                distance_all = {}

                for cen in center:
                    dis = np.linalg.norm(vector - cen)
                    distance_all[str(cen)] = dis

                distance_all = {k: v for k, v in sorted(distance_all.items(), key = lambda item: item[1])}
                nearest_cen = np.array(list(distance_all.keys())[0])

                for i in range(len(center)):
                    if center[i] is nearest_cen:
                        img.histogram[i] += 1

            if sum(img.histogram) != 0:
                img.histogram = img.histogram / sum(img.histogram)

        else:
            img.histogram = np.zeros(his_len)

    return imgs

def accuracy(test_imgs):
    ttl = 0
    correctness = 0
    for img in test_imgs:
        if img.label == img.predicted_label:
            ttl += 1
            correctness += 1
        else:
            ttl += 1

    score = float(correctness) / float(ttl)
    score = round(score, 3)

    return score

if __name__ == '__main__':

    train_images = []
    test_images = []
    label_name = []
    a_0 = 0  # TODO REMOVE
    idx_train = 0
    for dir_name_train in os.listdir('.\\data\\train'):
        if a_0 == 1:    # TODO REMOVE
            continue    # TODO REMOVE
        a_0 += 1        # TODO REMOVE
        a = 0           # TODO REMOVE
        for file_name_train in os.listdir('.\\data\\train\\' + dir_name_train):
            if a == 1:      # TODO REMOVE
                continue    # TODO REMOVE
            a += 1          # TODO REMOVE
            img_train = cv2.imread('.\\data\\train\\' + dir_name_train + '\\' + file_name_train)

            train_images.append(Image(img_train, os.path.splitext(file_name_train)[0],
                                      idx = idx_train, lab = dir_name_train))
            idx_train += 1
    b_0 = 0  # TODO REMOVE
    idx_test = 0
    for dir_name_test in os.listdir('.\\data\\validation'):
        if b_0 == 1:    # TODO REMOVE
            continue    # TODO REMOVE
        b_0 += 1        # TODO REMOVE
        b = 0           # TODO REMOVE
        for file_name_test in os.listdir('.\\data\\validation\\' + dir_name_test):
            if b == 1:      # TODO REMOVE
                continue    # TODO REMOVE
            b += 1          # TODO REMOVE
            img_test = cv2.imread('.\\data\\validation\\' + dir_name_test + '\\' + file_name_test)

            test_images.append(Image(img_test, os.path.splitext(file_name_test)[0],
                                     idx = idx_test, lab = dir_name_test))
            idx_test += 1

    train_images, train_images_key_des = update_images(train_images)
    test_images, test_images_key_des = update_images(test_images)

    descriptor = coalesce_des(train_images_key_des).astype('float32')

    words_num = 150
    compact, label, centroids = k_means(descriptor, words_num)

    train_images = histogram_made(train_images, centroids, words_num)
    test_images = histogram_made(test_images, centroids, words_num)

    knn_classifier = KnnClassifier(k = 1)
    test_images = knn_classifier.predict(train_images, test_images)
    # print(type(test_images[0].label))               # TODO REMOVE
    # print(type(test_images[0].predicted_label))     # TODO REMOVE

    accuracy_score = accuracy(test_images)
    print('Accuracy Score: ', end = '')
    print(accuracy_score)
