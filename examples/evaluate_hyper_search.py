import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import pickle
import collections

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


def plot_confusion_matrix(cm, cm_abs, out_path):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(cm, cmap=plt.cm.Blues, vmax=1, vmin=0)
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
    #tick_labels = ['resting', 'walking', 'eye_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming']
    tick_labels = ['resting', 'walking', 'head_grooming', 'foreleg_grooming', 'hind_grooming',]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=4, rotation=45)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=4, rotation=45)
    plt.savefig(out_path, transpartent=True, dpi=300)


dates = ["210830", "210910", "211026", "211027", "211029"]
flies = ["1", "2", "3", "2", "1"]
confusion_matrices = collections.defaultdict(list)
outdir = "cv_hyper_parameters_results"

for date, fly in zip(dates, flies):
    print(date)
    print("#" * 100)
    base_dir = f"my_results_angles_heur_new_and_old_data_full_videos_hyper_search_wo_{date}"
    for trained_model_dir in glob.glob(f"{base_dir}/multi-0/dtcn/test/version_*"):
        print(trained_model_dir)
        try:
            with open(os.path.join(trained_model_dir, "hparams.pkl"), "rb") as f:
                hyperparams = pickle.load(f)
        except Exception:
            continue
        lmbd_weak = hyperparams["lambda_weak"]
        lmbd_pred = hyperparams["lambda_pred"]

        model_param_file = os.path.join(trained_model_dir, "best_val_model.pt")

        expt_id = f"{date}_Ci1xG23_Fly{fly}_003_coronal"
        
        # where data is stored
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'my_data_full_videos')
        
        # DLC markers
        markers_file = os.path.join(data_dir, 'markers', expt_id + '_labeled.npy')
        # heuristic labels
        labels_file = os.path.join(data_dir, 'labels-heuristic', expt_id + '_labels.pkl')
        # hand labels
        hand_labels_file = os.path.join(data_dir, 'labels-hand', expt_id + '_labels.csv')
        
        # define data generator signals
        signals = ['markers', 'labels_weak', 'labels_strong']
        transforms = [ZScore(), None, None]
        paths = [markers_file, labels_file, hand_labels_file]
        
        # build data generator
        data_gen = DataGenerator([expt_id], [signals], [transforms], [paths], device="cuda", batch_size=hyperparams["batch_size"])
        
        hyperparams["batch_pad"] = 0
        model = Segmenter(hyperparams)
        model.to("cuda")
        model.load_state_dict(torch.load(model_param_file))
        
        #labels = np.genfromtxt(hand_labels_file, delimiter=",", dtype=np.int, encoding=None)
        #labels = labels[1:, 1:]
        labels = pd.read_csv(hand_labels_file, index_col=0)
        states = np.argmax(labels.values, axis=1)
        
        # get model predictions for each time point
        predictions = model.predict_labels(data_gen)["labels"][0]
        predictions = np.argmax(np.vstack(predictions), axis=1)
        
        scores = get_precision_recall(states, predictions, background=0)
        
        present_classes = set(states).intersection(set(predictions))
        present_classes.discard(0)
        present_classes = np.array(list(present_classes))
        #class_names = np.array(['resting', 'walking', 'eye_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming'])[present_classes - 1]
        class_names = np.array(['resting', 'walking', 'head_grooming', 'foreleg_grooming', 'hind_grooming', ])[present_classes - 1]
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
        
        plt.savefig(os.path.join(outdir, f"precision_recall_left_out_fly_lmbd_weak_{lmbd_weak}_lmbd_pred_{lmbd_pred}_{date}.png"))
        plt.close()

        obs_idxs = states != 0
        n_classes = 6
        cm = sklearn.metrics.confusion_matrix(states[obs_idxs], predictions[obs_idxs], labels=np.arange(1, n_classes), normalize="true")
        cm_abs = sklearn.metrics.confusion_matrix(states[obs_idxs], predictions[obs_idxs], labels=np.arange(1, n_classes))
        out_path = os.path.join(outdir, f"confusion_matrix_lmbd_weak_{lmbd_weak}_lmbd_pred_{lmbd_pred}_{date}.png")
        plot_confusion_matrix(cm, cm_abs, out_path)

        confusion_matrices[f"lmbd_weak_{lmbd_weak}_lmbd_pred_{lmbd_pred}"].append(cm_abs)

for key, matrices in confusion_matrices.items():
    cm_abs = np.sum(matrices, axis=0)
    cm = cm_abs / np.sum(cm_abs, axis=1)[:, np.newaxis]
    out_path = os.path.join(outdir, f"cv_confusion_matrix_lmbd_weak_{lmbd_weak}_lmbd_pred_{lmbd_pred}.png")
    plot_confusion_matrix(cm, cm_abs, out_path) 
