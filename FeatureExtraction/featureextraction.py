# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import tensorflow_hub as hub
from os import listdir, makedirs, remove
from os.path import isfile, join
import os


dataset = "../../datasets/core50_128x128"   # Dataset images
saveplace = "../../caracteristicas/Core50"  # Directory to save extracted features


__author__ = "Gabriel Vilariño Besteiro"


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
def pass_features(directory: str, directory_features: str, url_feature_extractor = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"):
    scenes = os.listdir(directory)
    create_directory(directory_features)

    feature_extractor_model = url_feature_extractor
    feature_extractor_layer = hub.KerasLayer(feature_extractor_model, input_shape=(128, 128, 3), trainable=False)

    print("[" + "".join(["-" for _ in scenes]) + "]")
    print("[", end="")
    for scene in scenes:
        objects = os.listdir(directory + "/" + scene)
        create_directory( directory_features + "/" + scene)
        for object in objects:
            create_directory( directory_features + "/" + scene + "/" + object)
            mypath = directory + "/" + scene + "/" + object

            images = []
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            for file in files:
                image = load_img(mypath + "/" + file)
                image = img_to_array(image)
                image = image / 255
                images.append(image)
            num_file=0
            for i in range(30):
                features = feature_extractor_layer(images[i * 10:(i + 1) * 10])
                for feature in features:
                    files [num_file] = files[num_file].replace(".png", "")
                    feature.numpy().tofile( directory_features + "/" + scene + "/" + object + "/" + files[num_file] + ".dat")
                    num_file=num_file+1
                # np.save("bbdd/x.dat", features, allow_pickle=False)
        print("-", end="")
    print("]")



# directory: String that contains the directory with the images that we want to process
# directory_features_ String that contains the new directory yhat will be created to save the features extracted
def pass_features_v1(directory: str, directory_features: str, url_feature_extractor = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"):
    scenes = os.listdir(directory)
    create_directory(directory_features)

    feature_extractor_model = url_feature_extractor
    feature_extractor_layer = hub.KerasLayer(feature_extractor_model, trainable=False)
    for scene in scenes:
        objects = os.listdir(directory + "/" + scene)
        create_directory( directory_features + "/" + scene)
        for object in objects:
            create_directory( directory_features + "/" + scene + "/" + object)
            mypath = directory + "/" + scene + "/" + object

            images = []
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            for file in files:
                image = load_img(mypath + "/" + file)
                image = img_to_array(image)
                image = tf.image.resize_with_pad(image, 224, 224)
                image = image / 255
                images.append(image)

            num_file=0
            for img in images:
                feature = feature_extractor_layer([img])
                files [num_file] = files[num_file].replace(".png", "")
                feature.numpy().tofile( directory_features + "/" + scene + "/" + object + "/" + files[num_file] + ".dat")
                num_file=num_file+1
                # np.save("bbdd/x.dat", features, allow_pickle=False)


pass_features(dataset, saveplace)