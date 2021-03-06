#!/usr/bin/env python

import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import scipy
from zlib import crc32
import os
import sys
import argparse


base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

index = {}

def _load_image(path):
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img


def _cosine_V(A,B):
    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)
    len_A = np.sqrt(A_f.power(2).sum())
    len_B = np.sqrt(B_f.power(2).sum())
    return 1 - (A_f.dot(B_f.transpose()) /(len_A * len_B)).data[0]

def _vectorizeImagesFromPaths(paths):
    X = [_load_image(img) for img in paths]

    features = model.predict(np.array(X))
    features = np.array([output.flatten() for output in features])
    intV = np.vectorize(np.uint16)
    features = intV(features * 65535 / np.max(features))
    features = scipy.sparse.csc_matrix(features)
    return features

###

def add_images(images, index):
    abs_paths = [os.path.abspath(img) for img in images]

    for i, path in enumerate(abs_paths):
        if path in index:
            if index[path]["crc32"] == crc32(open(path, 'rb').read()):
                abs_paths[i]=None # no need to add
            else:
                index.pop(path) # remove from index

    # Clean out abs_paths that were set to None
    abs_paths = [x for x in abs_paths if x != None]

    if len(abs_paths) == 0:
        return
    crc = [crc32(open(img, 'rb').read()) for img in abs_paths]

    features = _vectorizeImagesFromPaths(abs_paths)

    index.update({path: {"crc32": crc, "features": feature} for (path, crc, feature) in zip(abs_paths, crc, features) })
    return abs_paths


def load_index():
    if not os.path.exists('index.npz'):
        return {}
    index = np.load('index.npz', allow_pickle=True)['index']
    return index.tolist()

def save_index(index):
    np.savez_compressed('index.npz', index=index)


def query(path, index):
    queryFeatures = _vectorizeImagesFromPaths([path])

    result = []

    for k,v in index.items():
        indexFeatures = v['features']
        dist = _cosine_V(queryFeatures, indexFeatures)

        result.append([dist, k])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find similar images using CNN')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", dest='query', action='store_false')
    group.add_argument("--query", dest='query', action='store_true')
    parser.add_argument("images", nargs='*')
    args = parser.parse_args(sys.argv[1:])

    index = load_index()

    if len(args.images) == 0:
        if args.query:
            # Print all stored paths
            for path in index.keys():
                print(path)

    else:
        if args.query:
            query_results = query(args.images[0], index)
            query_results = sorted(query_results)
            for res in query_results:
                print("{:.2f} {}".format(*res))
        else:
            add_results = add_images(args.images, index)
            save_index(index)
            print(add_results)

