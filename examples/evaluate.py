import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from test_tube import HyperOptArgumentParser
import yaml
import torch
import pandas as pd
import sklearn.metrics

from daart.data import DataGenerator
from daart.eval import get_precision_recall, plot_training_curves
from daart.models import Segmenter
from daart.transforms import ZScore

import utils


def get_all_params():
    # raise error if user has other command line arguments specified
    if len(sys.argv[1:]) != 6:
        raise ValueError('No command line arguments allowed other than config file names')

    def add_to_parser(parser, arg_name, value):
        if arg_name == 'expt_ids':
            # treat expt_ids differently, want to parse full lists as one
            if isinstance(value, list):
                value = ';'.join(value)
            parser.add_argument('--' + arg_name, default=value)
        elif isinstance(value, list):
            parser.opt_list('--' + arg_name, options=value, tunable=True)
        else:
            parser.add_argument('--' + arg_name, default=value)

    # create parser
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--train_config', type=str)

    namespace, extra = parser.parse_known_args()

    # add arguments from all configs
    configs = [namespace.data_config, namespace.model_config, namespace.train_config]
    for config in configs:
        config_dict = yaml.safe_load(open(config))
        for (key, value) in config_dict.items():
            add_to_parser(parser, key, value)

    return parser.parse_args()

dates = ["210830", "210910", "211026", "211027", "211029"]
flies = ["1", "2", "3", "2", "1"]
for date, fly in zip(dates, flies):
    print(date)
    print("#" * 100)
    #trained_model_dir = f"my_results_angles_no_heur_new_data_full_videos_wo_{date}/multi-0/dtcn/test/version_0/"
    #trained_model_dir = f"my_results_angles_no_heur_new_and_old_data_full_videos_wo_{date}/multi-0/dtcn/test/version_0/"
    trained_model_dir = f"my_results_angles_heur_new__data_full_videos_wo_{date}/multi-0/dtcn/test/version_1/"
    model_param_file = os.path.join(trained_model_dir, "best_val_model.pt")
    batch_size = 500
    expt_id = f"{date}_Ci1xG23_Fly{fly}_003_coronal"
    
    # where data is stored
    base_dir = os.path.join(os.path.dirname(os.getcwd()), 'my_data_full_videos')
    
    # DLC markers
    markers_file = os.path.join(base_dir, 'markers', expt_id + '_labeled.npy')
    # heuristic labels
    labels_file = os.path.join(base_dir, 'labels-heuristic', expt_id + '_labels.pkl')
    # hand labels
    hand_labels_file = os.path.join(base_dir, 'labels-hand', expt_id + '_labels.csv')
    
    # define data generator signals
    signals = ['markers', 'labels_weak', 'labels_strong']
    transforms = [ZScore(), None, None]
    paths = [markers_file, labels_file, hand_labels_file]
    
    ## build data generator
    print(paths)
    print(batch_size)
    quit()
    data_gen = DataGenerator([expt_id], [signals], [transforms], [paths], device="cuda", batch_size=batch_size)
    
    hyperparams = vars(get_all_params())
    
    model = Segmenter(hyperparams)
    model.to("cuda")
    model.load_state_dict(torch.load(model_param_file))
    
    labels = np.genfromtxt(hand_labels_file, delimiter=",", dtype=np.int, encoding=None)
    labels = labels[1:, 1:]
    states = np.argmax(labels, axis=1)
    
    # get model predictions for each time point
    predictions = model.predict_labels(data_gen)["labels"][0]
    predictions = np.argmax(np.vstack(predictions), axis=1)
    
    scores = get_precision_recall(states, predictions, background=0)
    
    present_classes = set(states).intersection(set(predictions))
    present_classes.discard(0)
    present_classes = np.array(list(present_classes))
    class_names = np.array(['resting', 'walking', 'eye_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming'])[present_classes - 1]
    n_classes = len(class_names)
    
    # get rid of background class
    if len(scores["precision"]) != len(class_names):
        precision = scores["precision"][1:]
        recall = scores["recall"][1:]
    else:
        precision = scores["precision"]
        recall = scores["recall"]
    
    # plot precision and recall for each class
    plt.figure(figsize=(5, 5))
    for n, name in enumerate(class_names):
        plt.scatter(precision[n], recall[n], label=name)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()
    
    plt.savefig(os.path.join(trained_model_dir, "precision_recall_left_out_fly.png"))
    plt.close()

    obs_idxs = states != 0
    n_classes = 7
    cm = sklearn.metrics.confusion_matrix(states[obs_idxs], predictions[obs_idxs], labels=np.arange(1, n_classes), normalize="true")
    cm_abs = sklearn.metrics.confusion_matrix(states[obs_idxs], predictions[obs_idxs], labels=np.arange(1, n_classes))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(n_classes - 1):
        total = np.sum(cm_abs[i, :])
        for j in range(n_classes - 1):
            color = "black" if cm[i, j] < 0.5 else "white"
            ax.text(j, i, f"{cm[i, j] * 100:.2f}%\n({cm_abs[i, j]}/{total})",
                    va="center",
                    ha="center",
                    fontsize=5,
                    fontname="Arial",
                    color = color)
    ax.set_ylabel("Annotated behaviour", fontsize=6, fontname="Arial")
    ax.set_xlabel("Predicted behaviour", fontsize=6, fontname="Arial")
    tick_positions = np.arange(n_classes - 1)
    tick_labels = ['resting', 'walking', 'eye_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming']
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=4, rotation=45)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=4, rotation=45)
    plt.savefig(os.path.join(trained_model_dir, "confusion_matrix.png"), transpartent=True, dpi=300)
