import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import cv2
import glob
import os
import string
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC

# Returns images and labels of specified fruits
def getFruits(fruits, data_type):
    images = []
    labels = []
    dataType_path = "../fruits-360/" + data_type + "/"
    for i,f in enumerate(fruits):
        fruitFolder_path = dataType_path + f
        j=0
        for image_path in glob.glob(fruitFolder_path + "/*.jpg"):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)   
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(i)
            j+=1
        print("There are " , j , " " , data_type.upper(), " images of " , fruits[i].upper())
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# plots grid of images
def plot_image_grid(images, rows, cols, figsize=(15, 15)):
    figure, axes = plt.subplots(rows, cols, figsize=figsize)
    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            axes[i, j].axis('off')
            axes[i, j].imshow(images[n])
            n += 1

images_train, labels_train = getFruits(['Pineapple','Cocos','Corn','Dates'],'Training')
plot_image_grid(images_train[0:100], 10, 10)
plt.show()