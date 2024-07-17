import os
import pickle


def pickle_save(path, name, obj):
    with open(path + name, "wb") as f:
        pickle.dump(obj, f)
    return obj


def pickle_load(path, name):
    with open(path + name, "rb") as f:
        obj = pickle.load(f)
    return obj


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
