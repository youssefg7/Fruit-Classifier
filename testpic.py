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

# import image
RGB_img = cv2.imread("../fruits-360/Training/Pineapple/0_100.jpg")
RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2BGR)

# show original image
plt.imshow(RGB_img)

# splitting r g b channels
b, g, r = cv2.split(RGB_img)
print("Red channel")
print(type(r))
print(r.shape)
print("\nGreen channel")
print(type(g))
print(g.shape)
print("\nBlue channel")
print(type(b))
print(b.shape)

r_scaled = r/255
g_scaled = g/255
b_scaled = b/255

pca_r = PCA(0.9)
pca_r_trans = pca_r.fit_transform(r_scaled)
exp_var_r = pca_r.explained_variance_ratio_ * 100

pca_g = PCA(0.9)
pca_g_trans = pca_g.fit_transform(g_scaled)
exp_var_g = pca_r.explained_variance_ratio_ * 100

pca_b = PCA(0.9)
pca_b_trans = pca_b.fit_transform(b_scaled)
exp_var_b = pca_r.explained_variance_ratio_ * 100


pca_r_org = pca_r.inverse_transform(pca_r_trans)
pca_g_org = pca_g.inverse_transform(pca_g_trans)
pca_b_org = pca_b.inverse_transform(pca_b_trans)

img_compressed = cv2.merge((pca_b_org, pca_g_org, pca_r_org))

plt.figure(figsize=[7.3, 7.3])
plt.imshow(img_compressed)

cum_exp_var = np.cumsum(exp_var_r)

plt.figure(figsize=[7, 10])

plt.bar(range(1, len(exp_var_r)+1), exp_var_r, align='center',
        label='Individual explained variance')

plt.step(range(1, len(exp_var_r)+1), cum_exp_var, where='mid',
         label='Cumulative explained variance', color='red')
plt.ylabel('Explained variance percentage')
plt.xlabel('Principal component index')
plt.legend(loc='right')
plt.title("PCA for Red Color Channel")
plt.tight_layout()

plt.show()