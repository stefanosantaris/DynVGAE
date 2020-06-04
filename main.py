import argparse

from trainer import Trainer
from utils.dataset.DatasetDownloader import download_and_extract
import pickle as pkl
import os
from os import path

dataset_dict = {
    'large': {
        'url':'https://researchlab.blob.core.windows.net/datasets/LiveStream%20Datasets/LiveStream-16K.zip',
        'path':'data_large',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    },
    'medium': {
        'url':'https://researchlab.blob.core.windows.net/datasets/LiveStream%20Datasets/LiveStream-6K.zip',
        'path':'data_medium',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    },
    'small': {
        'url':'https://researchlab.blob.core.windows.net/datasets/LiveStream%20Datasets/LiveStream-4K.zip',
        'path':'data_small',
        'zip_file': 'dataset.zip',
        'extract_folder':'edgelists'
    }
}


def start_exp(args):
    trainer = Trainer(dataset_dict[args.dataset])
    results = trainer.train_model(args)
    if not path.exists('results'):
        os.mkdir('results')
    f = open("results/" +
             'dataset_' + str(args.dataset) +
             '_graph_' + str(args.start_graph) +
             '_emb_' + str(args.emb) +
             '_window_' + str(args.window) +
             '.pkl', "wb")
    pkl.dump(results, f)
    f.close()






if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='small',
                        help='Dataset File Name')
    parser.add_argument('--start_graph', type=int, default=0, help="Starting graph")
    parser.add_argument('--end_graph', type=int, default=7, help="Ending graph")
    parser.add_argument('--num_exp', type=int, default=1, help="Number of experiments")
    parser.add_argument('--emb', type=int, default=64, help="Embedding size")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--ns', type=int, default=1, help="Number of negative samples")
    parser.add_argument('--window', type=int, default=2, help="Window for evolution")
    parser.add_argument('--cuda', type=int, default=0, help="CUDA SUPPORT (0=FALSE/1=TRUE)")


    args = parser.parse_args()
    if args.window < 1:
        args.window = 1
    download_and_extract(dataset_dict[args.dataset])
    start_exp(args)






