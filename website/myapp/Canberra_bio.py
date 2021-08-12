import cv2 # Opencv
from pandas import read_csv
import numpy as np
from BiT import biodiversity
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt # to show query
import statistics
from statistics import mode
from scipy.spatial import distance
from typing import List
from os import listdir

def start():
    print("started")


path = "C:/Users/Zakaria/Documents/DECSession6/Stage/CBMIR/website/"

queryimg_path = path+"media/query/"
queryimg_dir:List[str] = listdir(queryimg_path)

images_path = path+"media/CRC/"
images_dir:List[str] = listdir(images_path)





def bio(file, dossier,nom):
    # Extract BiT features
    features = biodiversity(file)
    final_list = np.append(nom, features)
    final_list = np.append(final_list, dossier)
    return final_list



ListOfFeatures = list()




for fichier in listdir(queryimg_path):
    # Load image
    img_path = queryimg_path + fichier
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Feature extraction with BiT
    carac_bio = bio(img_gray, 'QueryImage',fichier)
    #print(carac_bio)
    # Create list of all the query
    ListOfFeatures.append(carac_bio)


def findimage(folder,name):
    path = "C:/Users/Zakaria/Documents/DECSession6/Stage/CBMIR/website/"
    images_path = path + "media/CRC/"
    images_dir: List[str] = listdir(images_path)
    for dossier in images_dir:
        dossier = str(dossier)
        folder = str(folder)
        if dossier == folder:
            for fichier in listdir(images_path + dossier + "/"):
                if fichier == name:
                    img_path = images_path + dossier + "/" + fichier
    return img_path


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = distance.canberra(test_row, pd.to_numeric(train_row[1:7]))
        distances.append((train_row, dist))

    def sortbydist(row):
        # print("row",row[1])
        return row[1]
    distances.sort(key=sortbydist)   #sort by distance ascending
    #print("dist",distances)
    neighbors = list()
    neighbors_classes = list()
    for i in range(num_neighbors):
        neighbors_classes.append(distances[i][0][8])  #classes

    def most_common(List):
        return max(set(List), key=List.count)

    common_class = most_common(neighbors_classes)  # returns the most common class of the query
    print("common class : ", common_class)

    for j in range(num_neighbors):
        cla_ss = distances[j][0][8]
        #if cla_ss == common_class:
        neighbors.append(distances[j][0]) # appends the train row to neighbors


    return neighbors










bit_file = path+'myapp/Crc_bio.csv'
bit = read_csv(bit_file,header=None)

array = bit.values

row0 = ListOfFeatures[0]


neighbors_count = 20  # how many query to return
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
    ax.set_title(" class "+ str(neighbor[8]), fontsize=7) # gives title to each image
    plt.margins(0, 0)
    print("i = ",i)
    i+= 1


plt.savefig(path+'media/output/crc_canberra_bio.png', bbox_inches='tight',pad_inches = 1)  #to save an image containing all the outputs
#img_color = cv2.imread(images_path+'crc_canberra_bio.png', 1)  # 1: Color image. 1 is optional.
#cv2.imshow("Outputs",img_color)  #show the outputs
#cv2.waitKey(0)








