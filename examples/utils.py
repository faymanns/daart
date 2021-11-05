import os.path
import itertools

def load_exp_dirs(path):
    dirs = []
    with open(path, "r") as f:
        for line in f:
            if line[:1] == "#":
                continue
            dirs.append(line.rstrip())
    return dirs


def get_trial_dir(date, genotype, fly, trial):
    date = int(date)
    fly = int(fly)
    trial = int(trial)
    locations = ["/mnt/lab_server/AYMANNS_Florian/Experimental_data/", "/mnt/data/FA/", "/mnt/data2/FA/"]
    imaging_types = ["coronal", "beh"]
    trial_dir = f"{date}_{genotype}/Fly{fly}/{trial:03d}_"
    possible_trial_dirs = [os.path.join(location, trial_dir + imaging_type) for location in locations for imaging_type in imaging_types]
    existance = [os.path.isdir(d) for d in possible_trial_dirs]
    if sum(existance) > 1:
        raise ValueError(f"Trial dirs for {trial_dir} exist on multiple modalities.")
    elif not any(existance):
        raise ValueError(f"Trial dir {trial_dir} does not exist.")
    else:
        return list(itertools.compress(possible_trial_dirs, existance))[0]


def get_fly_dir(directory):
    return os.path.dirname(directory.rstrip("/"))


def group_by_fly(dirs):
    groups = {}
    fly_dirs = list(map(get_fly_dir, dirs))
    for fly_dir in set(fly_dirs):
        groups[fly_dir] = [d for d in dirs if fly_dir in d]
    return groups


def _remove_storage_from_path(path):
    return path.replace("/mnt/data/FA/", "").replace("/mnt/lab_server/AYMANNS_Florian/Experimental_data/", "").replace("/mnt/data2/FA/", "")


def get_date(path):
    return int(_remove_storage_from_path(path).split("_")[0])


def get_fly_number(path):
    return int(path.split("Fly")[1].split("/")[0])


def get_trial_number(path):
    trial_dir_name = os.path.basename(os.path.normpath(path))
    trial_number = int(trial_dir_name.split("_")[0])
    return trial_number


def get_genotype(path):
    return "_".join(_remove_storage_from_path(path).split("/")[0].split("_")[1:])
