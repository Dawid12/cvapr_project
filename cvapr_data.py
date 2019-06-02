"version 0.1.1"  # by Krzysztof Lelonek 12.05.2019

# to import this module you should add "import sys" and "sys.path.append(PATH_TO_FOLDER_WITH_THIS_MODULE)"
#   statements at the beginning of your calling module
# it requires Python3 and MNE https://martinos.org/mne/stable/index.html , https://pypi.org/project/mne/
# everything that starts with _ is private, imports are imports, else is the module interface
#   you can check what those function do in their docs
# with default stimuli=254, each block seems to have at least 31606 samples (and it seems quite consistent),
#   didn't check 255, you can just take first 30k samples to always pass the same count

import os
import mne
import csv
import re

_env = {
    "repo_path": None,
    "bss": None,
}

def _emotion_class_to_int(emotion_class):
    if emotion_class == "Pos":
        return -1
    elif emotion_class == "Neg":
        return 0
    else:
        return 1

def _eval_path_from_eeg_path(eeg_path):
    eeg_filename = os.path.basename(eeg_path)
    patient_num = eeg_filename[4]
    session_num = eeg_filename[14]
    eval_path = os.path.join(_env['repo_path'],
                             *('enterface06_EMOBRAIN', 'Data', 'Common'),
                             f"Part{patient_num}SES{session_num}.log")
    return eval_path

def _load_block_ranges_from_marker(marker_path):
    current_line = ""
    block_ranges = []
    with open(marker_path, 'r') as marker_file:
        while not current_line.strip().endswith(f'"{_env["bss"]}"'):
            current_line = marker_file.readline()
        range_start = int(current_line.split("\t")[-2])  # [-1] is stimuli
        current_line = marker_file.readline()
        range_end = int(current_line.split("\t")[-2])
        block_ranges.append((range_start, range_end))
        for _1 in range(29):
            current_line = marker_file.readline()
            range_start = int(current_line.split("\t")[-2])
            current_line = marker_file.readline()
            range_end = int(current_line.split("\t")[-2])
            block_ranges.append((range_start, range_end))
    return block_ranges

def _load_evals(eval_path):
    current_line = ""
    eval_list = []
    with open(eval_path, 'r') as eval_file:
        for _1 in range(30):
            for _2 in range(5):
                current_line = eval_file.readline()
                if not current_line.strip():
                    current_line = eval_file.readline()
            line_parts = current_line.split(" ")
            eval_list.append( ( int(line_parts[-2][0]), int(line_parts[-1][0]) ) )
    return eval_list

def configure(repo_path, block_start_stimuli=254):
    """Sets up parameters needed for the module to work
    
    repo_path is the directory that containts the enterface06_EMOBRAIN directory (and everything else inside
    block_start_stimuli is the stimuli that indicates the beginning of the picture block
        - the other one is the end, it has to be either 254 or 255"""

    _env['repo_path'] = repo_path
    _env['bss'] = block_start_stimuli

def available_eeg_paths():
    """Returns list of paths for found eeg files
    
    The first patient data and second patient's first eeg are discarded"""

    eeg_dir_path = os.path.join(_env['repo_path'], *('enterface06_EMOBRAIN', 'Data', 'EEG'))
    eeg_dir_content_list = os.listdir(eeg_dir_path)
    eeg_file_list = filter(lambda el: el.endswith(".bdf"), eeg_dir_content_list)
    eeg_file_list = filter(lambda el: not el.startswith("Part1"), eeg_file_list)
    eeg_file_list = filter(lambda el: not el.startswith("Part2_IAPS_SES1"), eeg_file_list)
    eeg_file_path_list = [os.path.join(eeg_dir_path, eeg_file) for eeg_file in eeg_file_list]
    return eeg_file_path_list

def load_data_from_files(*file_numbers):
    '''Returns a list with each element corresponding to a picture block
    where each element is a tuple of two elements
    where first element is the EEG for the first channel
    and the second element is a tuple of two elements
    where first element is the patient arousal evaluation
    and second element is the patient valence evaluation
    (or may be other way around, it's just the way they are ordered in file)
    
    file_numbers are indexed from 0 (to 10)
    
    At the moment the eeg is only from the first channel'''

    emotion_classes = []
    with open(os.path.join(_env['repo_path'], 'enterface06_EMOBRAIN', 'Data', 'Common', 'IAPS_Classes_EEG_fNIRS.txt')) as emotion_classes_file:
        for emotion_class in csv.reader(emotion_classes_file, delimiter='\t'):
            emotion_classes.append(emotion_class)

    available_eeg_paths_snapshot = available_eeg_paths()
    eegs_to_load = [available_eeg_paths_snapshot[i] for i in file_numbers]
    markers_to_load = [single_eeg_to_load + '.mrk' for single_eeg_to_load in eegs_to_load]
    evals_to_load = [_eval_path_from_eeg_path(single_eeg_to_load)
                     for single_eeg_to_load in eegs_to_load]
    loaded_data = []
    for eeg_path, marker_path, eval_path in zip(eegs_to_load, markers_to_load, evals_to_load):
        block_ranges = _load_block_ranges_from_marker(marker_path)[:-1]  # list of tuples
        eval_list = _load_evals(eval_path)
        raw_eeg = mne.io.read_raw_edf(eeg_path, preload=True)
        raw_eeg = raw_eeg.pick_channels(raw_eeg.info["ch_names"][0:1])  # TODO

        if re.search("SES1", eeg_path):
            session = 0
        elif re.search("SES2", eeg_path):
            session = 1
        else:
            session = 2
        iterator = 0
        for (eeg_start, eeg_end), single_eval in zip(block_ranges, eval_list):
            block_eeg, block_times = raw_eeg[0, eeg_start:eeg_end + 1]
            block_eeg = block_eeg.flatten()
            emotion_class = emotion_classes[iterator][session]
            block_data = (block_eeg, single_eval, _emotion_class_to_int(emotion_class))
            loaded_data.append(block_data)
            iterator += 1
    return loaded_data
