import tensorflow as tf
import os


def download_and_extract(dataset_args):

    currentDirectory = os.getcwd()
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(dataset_args['url']),
                                                   origin=dataset_args['url'], extract=True, cache_subdir=currentDirectory + "/" + dataset_args['path'])
    print("Local copy of the dataset file: {}".format(train_dataset_fp))
