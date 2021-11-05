import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from test_tube import HyperOptArgumentParser
import yaml
import torch
import pandas as pd

from daart.data import DataGenerator
from daart.eval import get_precision_recall, plot_training_curves
from daart.models import Segmenter
from daart.transforms import ZScore

import utils


def hysteresis_filter(seq, n=5, n_false=None):
    """
    This function implements a hysteresis filter for boolean sequences.
    The state in the sequence only changes if n consecutive element are in a different state.

    Parameters
    ----------
    seq : 1D np.array of type boolean
        Sequence to be filtered.
    n : int, default=5
        Length of hysteresis memory.
    n_false : int, optional, default=None
        Length of hystresis memory applied for the false state.
        This means the state is going to change to false when it encounters
        n_false consecutive entries with value false.
        If None, the same value is used for true and false.

    Returns
    -------
    seq : 1D np.array of type boolean
        Filtered sequence.
    """
    if n_false is None:
        n_false = n
    seq = seq.astype(np.bool)
    state = seq[0]
    start_of_state = 0
    memory = 0

    current_n = n
    if state:
        current_n = n_false
    
    for i in range(len(seq)):
        if state != seq[i]:
            memory += 1
        elif memory < current_n:
            memory = 0
            continue
        if memory == current_n:
            seq[start_of_state:i - current_n + 1] = state
            start_of_state = i - current_n + 1
            state = not state
            if state:
                current_n = n_false
            else:
                current_n = n
            memory = 0
    seq[start_of_state:] = state
    return seq


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


skip_existing = False

for trial_dir in utils.load_exp_dirs("../my_data/trials.txt"):
    print(trial_dir)

    output_file = os.path.join(trial_dir, "behData/images/df3d/behaviour_predictions_daart.pkl")

    if skip_existing and os.path.isfile(output_file):
        print(f"skipping because {output_file} exists.")
        continue

    date = utils.get_date(trial_dir)
    genotype = utils.get_genotype(trial_dir)
    fly = utils.get_fly_number(trial_dir)
    trial = utils.get_trial_number(trial_dir)

    expt_id = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")

    # where data is stored
    base_dir = os.path.join(os.path.dirname(os.getcwd()), 'my_data')
    
    # where model results will be saved
    model_save_path = os.path.join(os.path.dirname(os.getcwd()), 'results')
    
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
    device = 'cuda'
    
    trial_splits = {
                    'train_tr': 9,
                    'val_tr': 1,
                    'test_tr': 0,
                    'gap_tr': 0
                   }
    
    # build data generator
    data_gen = DataGenerator([expt_id], [signals], [transforms], [paths], device=device, batch_size=500, trial_splits=trial_splits)
    
    hyperparams = vars(get_all_params())
    
    model = Segmenter(hyperparams)
    model.to(device)
    #model.load_state_dict(torch.load("my_results/multi-2/dtcn/test/version_0/best_val_model.pt"))
    model.load_state_dict(torch.load("my_results_8classes_5000/multi-0/dtcn/test/version_0/best_val_model.pt"))
    #model.load_state_dict(torch.load("my_results_with_heuristic_labels/multi-0/dtcn/test/version_1/best_val_model.pt"))
    
    # get model predictions for each time point
    predictions = np.vstack(model.predict_labels(data_gen)["labels"][0])
    
    class_names = np.array(['background', 'resting', 'walking', 'eye_grooming', 'antennal_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming'])#, 'backward_walking'])
    #class_names = np.array(['background', 'resting', 'walking', 'grooming', 'hindgrooming'])
    
    n_frames = predictions.shape[0]
    frames = np.arange(n_frames)
    indices = pd.MultiIndex.from_arrays(([date, ] * n_frames,
                                         [genotype, ] * n_frames,
                                         [fly, ] * n_frames,
                                         [trial, ] * n_frames,
                                         frames,
                                        ),
                                        names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
    prediction_df = pd.DataFrame(index=indices)
    prediction_df["Prediction"] = class_names[np.argmax(predictions, axis=1)]
    #prediction_df.loc[np.max(predictions, axis=1) < 0.75, "Prediction"] = ""

    for beh in class_names[1:]:
        binary_seq = prediction_df["Prediction"].values == beh
        #filtered_binary_seq = hysteresis_filter(binary_seq, n=100, n_false=50)
        filtered_binary_seq = hysteresis_filter(binary_seq, n=30)

        bool_index = np.logical_and(binary_seq, ~filtered_binary_seq)
        prediction_df.loc[bool_index, "Prediction"] = "background" 
        prediction_df.loc[filtered_binary_seq, "Prediction"] = beh

    prediction_df["Entropy"] = np.sum(-predictions * np.log(predictions), axis=1)

    for i, beh in enumerate(class_names):
        prediction_df[f"Probability {beh}"] = predictions[:, i]

    prediction_df.to_pickle(output_file)
    
