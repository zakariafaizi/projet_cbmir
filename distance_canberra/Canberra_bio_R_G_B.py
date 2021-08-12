import cv2 # Opencv
from pandas import read_csv
import numpy as np
from BiT import bio_taxo,biodiversity,taxonomy
from paths import queryimg_path, queryimg_dir,outex_path, outex_dir, output_path
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt # to show query
import statistics
from statistics import mode
from scipy.spatial import distance




def bio(file, dossier,nom):
    # Extract BiT features
    b,g,r = cv2.split(file)
    file = b+g+r
    features = biodiversity(file)
    final_list = np.append(nom, features)
    final_list = np.append(final_list, dossier)
    return final_list



def sortbydist(row):
    #print("row",row[1])
    return row[1]   #sort by distance ascending


def most_common(List):
    return max(set(List), key = List.count)



ListOfFeatures = list()



for dossier in queryimg_dir:
    # print(dossier)
    for fichier in listdir(queryimg_path + dossier + "/"):
        #print("Dossier: " + dossier + "- Image: " + fichier)
        # Load image
        img_path = queryimg_path + dossier + "/" + fichier
        img = cv2.imread(img_path)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Feature extraction with BiT
        carac_bio = bio(img, dossier,fichier)
        print(carac_bio)
        # Create list of all the query
        ListOfFeatures.append(carac_bio)


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
        dist = distance.canberra(test_row, pd.to_numeric(train_row[1:7]))
        distances.append((train_row, dist))
    distances.sort(key=sortbydist)   #sort by distance ascending
    print("dist",distances)
    neighbors = list()
    neighbors_classes = list()
    for i in range(num_neighbors):
        neighbors_classes.append(distances[i][0][8])  #classes

    common_class = most_common(neighbors_classes)  # returns the most common class of the query
    print("common class : ", common_class)

    for j in range(num_neighbors):
        cla_ss = distances[j][0][8]
        #if cla_ss == common_class:
        neighbors.append(distances[j][0]) # appends the train row to neighbors


    return neighbors










bit_file = '../datasets/Crc_bio_r_g_b.csv'
bit = read_csv(bit_file,header=None)

array = bit.values

row0 = ListOfFeatures[0]


neighbors_count = 50  # how many query to return
neighbors = get_neighbors(array,pd.to_numeric(row0[1:7]), neighbors_count)




fig = plt.figure(figsize=(10,20))
i = 0
for neighbor in neighbors:
    print("Result Image's name : ", neighbor[0], "at folder : ", neighbor[8])
    imgpath = findimage(neighbor[8], neighbor[0])
    img_color = cv2.imread(imgpath, 1)  # 1: Color image. 1 is optional.
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    ax =fig.add_subplot(neighbors_count, 5,i+1)  # 10 rows and 1 column
    plt.imshow(img_color)
    plt.axis('off')
    ax.set_title("class "+ str(neighbor[8]), fontsize=7) # gives title to each image
    plt.margins(0, 0)
    print("i = ",i)
    i+= 1


plt.savefig(output_path+'crc_canberra_bio_r_g_b.png', bbox_inches='tight',pad_inches = 1)  #to save an image containing all the outputs
img_color = cv2.imread(output_path+'crc_canberra_bio_r_g_b.png', 1)  # 1: Color image. 1 is optional.
cv2.imshow("Outputs",img_color)  #show the outputs
cv2.waitKey(0)








