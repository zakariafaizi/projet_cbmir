import cv2 # Opencv
from pandas import read_csv
import numpy as np
from BiT import bio_taxo,biodiversity,taxonomy
from paths import queryimg_path, queryimg_dir,outex_path, outex_dir
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt # to show images





# Define BiT features extraction
def BiT(file, dossier):
    # Extract BiT features
    features = bio_taxo(file)
    final_list = np.append(features,dossier)
    return final_list

def BiT_bio(file, dossier):
    # Extract BiT features
    features = biodiversity(file)
    final_list = np.append(features,dossier)
    return final_list

def BiT_taxo(file, dossier):
    # Extract BiT features
    features = taxonomy(file)
    final_list = np.append(features,dossier)
    return final_list


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


ListOfFeatures = list()



for dossier in queryimg_dir:
    # print(dossier)
    for fichier in listdir(queryimg_path + dossier + "/"):
        #print("Dossier: " + dossier + "- Image: " + fichier)
        # Load image
        img_path = queryimg_path + dossier + "/" + fichier
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Feature extraction with BiT
        carac_bit = BiT(img_gray, dossier)
        print(carac_bit)
        # Create list of all the images
        ListOfFeatures.append(carac_bit)


def findimage(folder,name):
    for dossier in outex_dir:
        dossier = str(dossier)
        folder = str(folder)
        if dossier == folder:
            for fichier in listdir(outex_path + dossier + "/"):
                if fichier == name:
                    img_path = outex_path + dossier + "/" + fichier
    return img_path


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, pd.to_numeric(train_row[1:14]))
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


bit_file = 'Outex_BiT_names.csv'
bit = read_csv(bit_file,header=None)

array = bit.values

row0 = ListOfFeatures[0]


neighbors_count = 10   # how many images to return
neighbors = get_neighbors(array,pd.to_numeric(row0[0:14]), neighbors_count)



f, images = plt.subplots(neighbors_count,1)  #10 rows and 1 column

i = 0
for neighbor in neighbors:
    print("Result Image's name : ", neighbor[0], "at folder : ", neighbor[15])
    imgpath = findimage(neighbor[15], neighbor[0])
    img_color = cv2.imread(imgpath, 1)  # 1: Color image. 1 is optional.
    images[i].imshow(img_color)
    i+= 1
    #cv2.waitKey(0)
plt.show()







