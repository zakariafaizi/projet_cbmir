import cv2 # Opencv
import numpy as np
from BiT import bio_taxo,biodiversity,taxonomy
from paths import outex_path, outex_dir
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt
from mahotas import features as harl  #haralick
from skimage.feature import greycomatrix, greycoprops






def bit_haralick(file, dossier,nom):
    # Extract BiT features
    features = bio_taxo(file)
    final_list = np.append(nom, features)
    # Extract haralick features
    features2 = harl.haralick(file).mean(0)
    final_list = np.append(final_list, features2)
    final_list = np.append(final_list, dossier)
    return final_list

#GLCM funct
def bit_glcm(file,dossier,nom):
    # Extract BiT features
    features = bio_taxo(file)
    final_list = np.append(nom, features)
    glcm = greycomatrix(file, [5], [0], 256, symmetric=True, normed=True)
    #params : ( image , distances , angles , levels , symmetric => default False , normalization => default False  )
    diss = np.float32(greycoprops(glcm, 'dissimilarity')[0,0])
    cont = np.float32(greycoprops(glcm, 'contrast')[0, 0])
    corr = np.float32(greycoprops(glcm, 'correlation')[0, 0])
    ener = np.float32(greycoprops(glcm, 'energy')[0, 0])
    homo = np.float32(greycoprops(glcm, 'homogeneity')[0, 0])
    glcm_features = [diss,cont,corr,ener,homo]

    final_list = np.append(final_list, glcm_features)
    final_list = np.append(final_list, dossier)
    return final_list



ListOfFeatures = list()
initial = dt.datetime.now()

iteration = 0
for dossier in outex_dir:
    # print(dossier)
    for fichier in listdir(outex_path + dossier + "/"):
        #print("Dossier: " + dossier + "- Image: " + fichier)
        # Load image
        img_path = outex_path + dossier + "/" + fichier
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Feature extraction with BiT
        #carac_bit_haralick = bit_haralick(img_gray, dossier, fichier)
        carac_bit_glcm = bit_glcm(img_gray, dossier, fichier)
        #print(carac_bit_haralick)
        print(carac_bit_glcm)
        #ListOfFeatures.append(carac_bit_haralick)
        ListOfFeatures.append(carac_bit_glcm)

    iteration += 1

# Create dataframe
df_final = pd.DataFrame.from_records(data=ListOfFeatures)
# Create cv file from dataframe
df_final.to_csv('Outex_Bit_Glcm.csv', sep=',', index=False, header=False)

print('It took :', dt.datetime.now() - initial)