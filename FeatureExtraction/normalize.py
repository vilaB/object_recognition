# import the necessary packages
import numpy as np
from sklearn.preprocessing import normalize
from os import listdir, makedirs, remove
from os.path import isfile, join
import os


dataset = "../../dataset/Core50"                   # Raíz dataset
saveplace = "../../dataset/Core50_norm"            # Punto de guardado

def create_directory(directory):
    try:
        makedirs(directory)
    except OSError as e:
        if False:
            for file in listdir(directory):
                if isfile(directory + "/" + file):
                    remove(directory + "/" + file)


# directory: String that contains the directory with the images that we want to process
# directory_features_ String that contains the new directory yhat will be created to save the features extracted
def pass_features(directory: str, directory_features_norm: str):
    scenes = os.listdir(directory)
    create_directory(directory_features_norm)

    print("[" + "".join(["-" for _ in scenes]) + "]")
    print("[", end="")
    for scene in scenes:
        objects = os.listdir(directory + "/" + scene)
        create_directory( directory_features_norm + "/" + scene)
        for object in objects:
            create_directory( directory_features_norm + "/" + scene + "/" + object)
            mypath = directory + "/" + scene + "/" + object
            
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            for file in files:
                x = np.fromfile(mypath + "/" + file, dtype=np.float32)
                norm = normalize(x[:,np.newaxis], axis=0).ravel()
                np.save(directory_features_norm + "/" + scene + "/" + object + "/" + file.replace(".dat", ""), norm)
        print("-", end="")
    print("]")

pass_features(dataset, saveplace)