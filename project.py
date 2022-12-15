import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import cv2
import glob
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
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# import images
def getFruits(fruits):
    images = []
    labels = []
    r_components = []
    g_components = []
    b_components = []
        
    for i,f in enumerate(fruits):
        fruitFolder_path = "../fruits-360/" + "*" + "/" + f
        j=0
        for image_path in glob.glob(fruitFolder_path + "/*.jpg"):
            image = cv2.imread(image_path)   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            b , g , r = cv2.split(image)
            images.append(image)
            labels.append(fruits[i])
            r_components.append(r/255)
            g_components.append(g/255)
            b_components.append(b/255)
            j+=1
        print("There are " , j, " images of " , fruits[i].upper())
    images = np.array(images)
    labels = np.array(labels)
    r_components = np.array(r_components)
    g_components = np.array(g_components)
    b_components = np.array(b_components)
    return images, labels, r_components, g_components, b_components


# Plot images grid
def plot_image_grid(images, nb_rows, nb_cols, figsize=(15, 15)):
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    n = 0
    if(nb_rows == 1 or nb_cols ==1):
            for i in range(0,nb_cols+nb_rows-1):
                axs[i].axis('off')
                axs[i].imshow(images[n])
                n += 1
    else:
        for i in range(0, nb_rows):
            for j in range(0, nb_cols):
                axs[i, j].axis('off')
                axs[i, j].imshow(images[n])
                n += 1

# get fruits
images_X, labels_Y, r_components, g_components, b_components = getFruits(["Pepino","Pear Williams"])

# plot_image_grid(images_train[0:100],10,10)


# PCA
def performPCA(n_components, c_components):
    pca = PCA(n_components)
    pca_trans = pca.fit_transform(c_components.reshape(len(c_components),100*100))
    pca_projected = pca.inverse_transform(pca_trans)
    return pca_trans, pca_projected

b_reduced_flat, b_reduced_projected = performPCA(10, b_components) 
g_reduced_flat, g_reduced_projected = performPCA(10, g_components) 
r_reduced_flat, r_reduced_projected = performPCA(10, r_components) 


# Classification preparation
def prepareForTraining(b_reduced_flat,g_reduced_flat,r_reduced_flat):
    
    reduced_flat_imgs = []

    for i in range(0, len(r_reduced_flat)):
        img_flat = np.append(np.append(b_reduced_flat[i],g_reduced_flat[i]),r_reduced_flat[i])
        reduced_flat_imgs.append(img_flat)
    
    reduced_flat_imgs = np.array(reduced_flat_imgs)
    return reduced_flat_imgs

X_reduced_flat = prepareForTraining(b_reduced_flat,g_reduced_flat,r_reduced_flat)
x_train, x_test, y_train, y_test = train_test_split(X_reduced_flat,labels_Y,test_size=0.40)
print('Splitted Successfully')

#SVM
svm = SVC(kernel="linear", gamma='auto', probability=True)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
print(f"The model is {metrics.accuracy_score(y_pred,y_test)*100}% accurate")



plt.show()


