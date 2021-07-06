import cv2 # Opencv
import numpy as np
from BiT import bio_taxo
from paths import outex_path, outex_dir
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt




# Define BiT features extraction
def BiT(file, dossier,nom):
    # Extract BiT features
    features = bio_taxo(file)
    final_list = np.append(nom,features)
    final_list = np.append(final_list,dossier)
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
        carac_bit = BiT(img_gray, dossier, fichier)
        print(carac_bit)
        # Create list of all the images
        ListOfFeatures.append(carac_bit)
    iteration += 1

# Create dataframe
df_final = pd.DataFrame.from_records(data=ListOfFeatures)
# Create cv file from dataframe
df_final.to_csv('Outex_BiT_names.csv', sep=',', index=False, header=False)

print('It took :', dt.datetime.now() - initial)