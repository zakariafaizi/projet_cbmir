import cv2 # Opencv
import numpy as np
from BiT import bio_taxo,biodiversity,taxonomy
from paths import outex_path, outex_dir
import pandas as pd
from typing import List
from os import listdir
import datetime as dt
from math import sqrt





def bio(file, dossier,nom):
    # Extract Bio features
    b,g,r = cv2.split(file)
    file = r+g+b
    print(b,g,r)
    features = biodiversity(file)
    final_list = np.append(nom, features)
    final_list = np.append(final_list, dossier)
    return final_list




ListOfFeatures = list()
initial = dt.datetime.now()


for dossier in outex_dir:
    # print(dossier)
    for fichier in listdir(outex_path + dossier + "/"):
        # Load image
        img_path = outex_path + dossier + "/" + fichier
        img = cv2.imread(img_path)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        carac_bio = bio(img, dossier, fichier)
        print(carac_bio)
        # Create list of all the query
        ListOfFeatures.append(carac_bio)


# Create dataframe
df_final = pd.DataFrame.from_records(data=ListOfFeatures)
# Create cv file from dataframe
df_final.to_csv('../datasets/Crc_bio_r_g_b.csv', sep=',', index=False, header=False)

print('It took :', dt.datetime.now() - initial)