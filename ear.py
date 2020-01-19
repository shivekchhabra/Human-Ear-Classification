"""
@author: Shivek
"""
import cv2
import numpy as np
import matplotlib.image as mimg
import os
from sklearn import svm, metrics
from skimage.feature import hog, daisy
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def hog_feature_extraction():
    data = []
    label = []
    print('Using Hog:')
    for i in range(1, 11):
        path = cwd + ('/datasets/u%d' % (i))
        all_path = glob.glob(os.path.join(path, '*.bmp'))
        for j in all_path:
            im = mimg.imread(j)
            des, feat = hog(im, orientations=8, visualize=True)
            # cv2.imshow('img',im)
            # cv2.imshow('img2',feat)
            # cv2.waitKey()
            data.append(feat.reshape(1, -1))
            label.append(i)

    x = np.array(data, dtype='float32').reshape(100, 55488)
    y = np.array(label, dtype='float32')

    return x, y


def svm_classifier(x_train, y_train, x_test, y_test):
    svm_model = svm.SVC(kernel='linear')
    y_pred = svm_model.fit(x_train, y_train).predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc * 100


def random_forest_classifier(x_train, y_train, x_test, y_test):
    rf_model = RandomForestClassifier(n_jobs=2, criterion='entropy', n_estimators=55,
                                      random_state=23)
    y_pred = rf_model.fit(x_train, y_train).predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc * 100


def daisy_feature_extraction():
    print('Using Daisy:')
    data = []
    label = []
    for i in range(1, 11):
        path = cwd + ('/datasets/u%d' % (i))
        all_path = glob.glob(os.path.join(path, '*.bmp'))
        for j in all_path:
            im = mimg.imread(j)
            desc, feat = daisy(im, step=180, radius=58, rings=2, histograms=6,
                               orientations=8, visualize=True)
            # cv2.imshow('img',im)
            # cv2.imshow('img2',feat)
            # cv2.waitKey()
            data.append(feat.reshape(1, -1))
            label.append(i)
    x = np.array(data, dtype='float32').reshape(100, 166464)
    y = np.array(label, dtype='float32')
    return x, y


def run_hog():
    x, y = hog_feature_extraction()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=47, test_size=0.3)
    acc_svm = svm_classifier(x_train, y_train, x_test, y_test)
    print('Accuracy from SVM using HOG feature extraction is:- ', acc_svm)
    acc_rf = random_forest_classifier(x_train, y_train, x_test, y_test)
    print('Accuracy from Random Forest using Hog feature extraction is:- ', acc_rf)


def run_daisy():
    x, y = daisy_feature_extraction()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=47, test_size=0.3)
    acc_svm = svm_classifier(x_train, y_train, x_test, y_test)
    print('Accuracy from SVM using Daisy feature extraction is:- ', acc_svm)
    acc_rf = random_forest_classifier(x_train, y_train, x_test, y_test)
    print('Accuracy from Random Forest using Daisy feature extraction is:- ', acc_rf)


if __name__ == '__main__':
    cwd = os.getcwd()
    run_hog()
    print('\n\n')
    run_daisy()
