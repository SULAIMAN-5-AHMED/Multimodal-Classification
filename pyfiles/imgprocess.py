"""
penumonia = {normal = 0, penumonia = 1 }
lung cancer = {normal: 0,Maligant: 1, Bengin: 2}
brain tumor = {no tumor :0, glioma: 1, meningioma: 2, pituitary: 3}
"""

import pandas as pd
import os
import numpy as np
import cv2 as cv


def get_data(path_dir):
    img_path =[]
    label =[]
    for folder in os.listdir(path_dir):
        for img in os.listdir(os.path.join(path_dir, folder)):
            img_path.append(img)
            label.append(folder)
    data = {"img_path":img_path, "label":label}
    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
data = get_data('../Datasets/LungCancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Test')
print(data.head(5))

images = []
labels = []
n = 0
for index , row in data.iterrows():

    img_path = row["img_path"]
    label = row["label"]
    img = cv.imread(r'../Datasets/LungCancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Test/{}/{}'.format(label,img_path))
    max_pixel = 255.0
    img = cv.cvtColor(cv.resize(img,(224,224)), cv.COLOR_BGR2RGB)
    img_norm = img.astype("float32")/max_pixel
    print(f"processing image {n} with shape: {img_norm.shape}")
    images.append(np.array(img_norm))
    labels.append(int(label))
    n += 1

np.save("../Datasets/LungCancer/LungcancerTestX.npy", np.array(images))
np.save("../Datasets/LungCancer/LungcancerTestY.npy", np.array(labels))



