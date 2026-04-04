"""
penumonia = {normal = 0, penumonia = 1 }
lung cancer = {normal: 0,cancer: 1}
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
data = get_data('C:\\Users\\sulai\\Desktop\\Python\\MedScan\\Datasets\\penumonia\\test')
print(data.head(5))

images = []
labels = []
n = 0
for index , row in data.iterrows():

    img_path = row["img_path"]
    label = row["label"]
    img = cv.imread(r'C:\\Users\\sulai\\Desktop\\Python\\MedScan\\Datasets\\penumonia\\test\\{}\\{}'.format(label,img_path))
    max_pixel = 255.0
    img = cv.cvtColor(cv.resize(img,(224,224)), cv.COLOR_BGR2RGB)
    img_norm = img.astype("float32")/max_pixel
    print(f"frocessing image {n} with shape: {img_norm.shape}")
    images.append(np.array(img_norm))
    labels.append(int(label))
    n += 1

np.save("../Datasets/penumonia/penumonia224testX.npy", np.array(images))
np.save("../Datasets/penumonia/penumoniatestY.npy", np.array(labels))



