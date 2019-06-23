"version 0.3.2"  # by Krzysztof Lelonek


'''
This module allows loading the data from the enterface06_EMOBRAIN database.
It requires Python3 and MNE https://martinos.org/mne/stable/index.html , https://pypi.org/project/mne/.
Everything that starts with _ is private, imports are imports, else is the module interface,
you can check what those function do in their docs.


FILTERING AND POWER SPECTRA

Basic usage is to simply use load_data_from_files(numbers...) to get list of EEG samples for all blocks.

Then, each of those picture block data objects can be filtered,
or the data can also be filtered while loading (if you add keyword parameters to load_data_from_files).
In the first case, the data is first split into blocks and then each block is filtered,
in the second case, the data is first filtered, then split into blocks,
and the resulting samples will be different in both cases.

You can also get the power spectra from the EEG, then you always have to define range of frequencies.
Summing up, you can:
    - load blocks and do power spectra on each
    - load blocks, filter each block and do power spectra on each
    - load blocks, filtering whole file while loading, and do power spectra on each
    - get power spectra for blocks directly from each EEG
    - filter whole EEG and then get power spectra for blocks directly from it
and in each case the resulting power spectra may be, sometimes even very, different.
'''


import os
import mne
import numpy as np
from mne.time_frequency import psd


_env = {
    "repo_path": None,
    "bss": None,
}


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


class PictureBlockData:

    '''Represents data gathered during one picture block'''

    def __init__(self, single_eval, block_eeg, low_freq = None, high_freq = None):
        '''Constructs PictureBlockData object based on data gathered from a picture block

        Parameters:
            - single_eval - tuple of patient's arousal and valence evaluation
                (or may be other way around, it's just the way they are ordered in file)
            - block_eeg - EEG gathered during the picture block, provided by mne library
            - low_freq - low frequency at which the signal was filtered, in Hz
                or None in case of no filtering
            - high_freq - low frequency at which the signal was filtered, in Hz
                or None in case of no filtering
        '''
        self._low_freq = low_freq
        self._high_freq = high_freq
        self._mne_eeg = block_eeg
        self._evaluation = single_eval

    @property
    def mne_eeg(self):
        '''Wrapped mne EEG object'''
        return self._mne_eeg

    @property
    def evaluation(self):
        '''Tuple of patient's arousal and valence evaluation
        (or may be other way around, it's just the way they are ordered in file)'''
        return self._evaluation

    @property
    def raw_eeg(self):
        '''Samples of patient's EEG,
        It is a np.array, with first dimension being channel number, and second - sample number'''
        return self.mne_eeg[:,:][0]

    def power_spectrum(self, low_freq = None, high_freq = None, step = 1):
        '''Returns a tuple,
        where first element is a np.array
        where i-th element is the power spectrum data for i-th frequency
        and second element is a np.array
        where i-th element is the i-th frequency, in Hz.
        
        The frequencies will be from low_freq to high_freq, inclusive, with step step.'''
        if low_freq is None:
            low_freq = self._low_freq if self._low_freq is not None else 0
        if high_freq is None:
            high_freq = self._high_freq if self._high_freq is not None else float("inf")
        sample_freq = self.mne_eeg.info["sfreq"]
        return psd.psd_welch(self.mne_eeg, low_freq, high_freq, n_fft = round(sample_freq / step))

    def copy(self):
        '''Creates and returns a copy of the object'''
        return PictureBlockData(self.single_eval, self.mne_eeg.copy(), self._low_freq, self._high_freq)
    
    def filter(self, low_freq = None, high_freq = None):
        '''Filters the EEG between low_freq and high_freq (in Hz, can be None).
        Modifies the object in place, use copy first to preserve initial object.
        Return the object after modification.'''
        self.mne_eeg.filter(low_freq, high_freq)
        self._low_freq = low_freq if low_freq is not None else self._low_freq
        self._high_freq = high_freq if high_freq is not None else self._high_freq
        return self


def configure(repo_path, block_start_stimuli=254):
    """Sets up parameters needed for the module to work
    
    repo_path is the directory that containts the enterface06_EMOBRAIN directory (and everything else inside)
    block_start_stimuli is the stimuli that indicates the beginning of the picture block
        - the other one is the end, it has to be either 254 or 255"""

    _env['repo_path'] = repo_path
    _env['bss'] = block_start_stimuli


def available_eeg_paths():
    """Returns list of paths for found EEG files
    
    The first patient data and second patient's first EEG are discarded"""

    eeg_dir_path = os.path.join(_env['repo_path'], *('enterface06_EMOBRAIN', 'Data', 'EEG'))
    eeg_dir_content_list = os.listdir(eeg_dir_path)
    eeg_file_list = filter(lambda el: el.endswith(".bdf"), eeg_dir_content_list)
    eeg_file_list = filter(lambda el: not el.startswith("Part1"), eeg_file_list)
    eeg_file_list = filter(lambda el: not el.startswith("Part2_IAPS_SES1"), eeg_file_list)
    eeg_file_path_list = [os.path.join(eeg_dir_path, eeg_file) for eeg_file in eeg_file_list]
    return eeg_file_path_list


def load_data_from_files(*file_numbers, low_freq = None, high_freq = None):
    '''Returns a list with PictureBlockData objects
    where each element corresponds to one picture block.
    
    Parameters:
        - file_numbers - indices of available EEG files (from 0 to 10).
            There are loaded all blocks for each number in the order their indices are passed,
            i.e. load_data_from_files(3,1,2,3) will first return all blocks
            from file of index 3, then index 1, 2 and 3 again (duplicated).
        - low_freq - low frequency used for filtering, in Hz, or None
        - high_freq - high frequency used for filtering, in Hz, or None
    
    Notice that low_freq and high_freq are keyword-only parameters.
    Loading data without filtering should be faster.'''

    available_eeg_paths_snapshot = available_eeg_paths()
    eegs_to_load = [available_eeg_paths_snapshot[i] for i in file_numbers]
    markers_to_load = [single_eeg_to_load + '.mrk' for single_eeg_to_load in eegs_to_load]
    evals_to_load = [_eval_path_from_eeg_path(single_eeg_to_load)
                     for single_eeg_to_load in eegs_to_load]
    no_filtering = low_freq is None and high_freq is None
    loaded_data = []
    for eeg_path, marker_path, eval_path in zip(eegs_to_load, markers_to_load, evals_to_load):
        block_ranges = _load_block_ranges_from_marker(marker_path)  # list of tuples
        eval_list = _load_evals(eval_path)
        raw_eeg = mne.io.read_raw_edf(eeg_path)
        if no_filtering:
            pass
        else:
            raw_eeg.load_data().pick_channels(raw_eeg.info["ch_names"][:-8]).filter(low_freq, high_freq)
        for (eeg_start, eeg_end), single_eval in zip(block_ranges, eval_list):
            if no_filtering:
                sample_freq = raw_eeg.info["sfreq"]
                block_eeg = raw_eeg.copy().crop(eeg_start / sample_freq, eeg_end / sample_freq).load_data().pick_channels(raw_eeg.info["ch_names"][:-8])
            else:
                block_eeg = raw_eeg.copy().crop(raw_eeg.times[eeg_start], raw_eeg.times[eeg_end])
            picture_block_data = PictureBlockData(single_eval, block_eeg, low_freq, high_freq)
            loaded_data.append(picture_block_data)
    return loaded_data


def load_power_spectra_from_files(*file_numbers, low_freq = None, high_freq = None, step = 1, filter = False):
    '''Returns a list,
    where each element is a tuple corresponding to a picture block,
    where first element is a tuple,
    where first element is a np.array
    where i-th element is the power spectrum data for i-th frequency
    and second element is a np.array
    where i-th element is the i-th frequency, in Hz
    and second element is a tuple
    of patient's arousal and valence evaluation
    (or may be other way around, it's just the way they are ordered in file).

    Visually:
    [ (   ([power1, power2, ...], [freq1, freq2, ...]), (arousal, valence)   ),
      (   ([power1, power2, ...], [freq1, freq2, ...]), (arousal, valence)   ),
      ...
                                                                                ]
    Parameters:
        - file_numbers - indices of available EEG files (from 0 to 10).
            There are loaded all blocks for each number in the order their indices are passed,
            i.e. load_data_from_files(3,1,2,3) will first return data for all blocks
            from file of index 3, then index 1, 2 and 3 again (duplicated).
        - low_freq - low frequency used for power spectra calcuation, in Hz
        - high_freq - high frequency used for power spectra calcuation, in Hz
        - step - the frequency step in returned power spectrum
        - filter - if True, the data will be explicitly filtered before power spectra calculation
            between the given frequencies
    
    This function uses mne function to get power spectra directly from whole EEG while defining frequencies.
    Notice that low_freq and high_freq are keyword-only parameters.'''

    available_eeg_paths_snapshot = available_eeg_paths()
    eegs_to_load = [available_eeg_paths_snapshot[i] for i in file_numbers]
    markers_to_load = [single_eeg_to_load + '.mrk' for single_eeg_to_load in eegs_to_load]
    evals_to_load = [_eval_path_from_eeg_path(single_eeg_to_load)
                     for single_eeg_to_load in eegs_to_load]
    loaded_data = []
    for eeg_path, marker_path, eval_path in zip(eegs_to_load, markers_to_load, evals_to_load):
        block_ranges = _load_block_ranges_from_marker(marker_path)  # list of tuples
        eval_list = _load_evals(eval_path)
        raw_eeg = mne.io.read_raw_edf(eeg_path, preload = True)
        raw_eeg.pick_channels(raw_eeg.info["ch_names"][:-8])
        if filter and not (low_freq is None and high_freq is None):
            raw_eeg.filter(low_freq, high_freq)
        low_freq = low_freq if low_freq is not None else 0
        high_freq = high_freq if high_freq is not None else float("inf")
        for (eeg_start, eeg_end), single_eval in zip(block_ranges, eval_list):
            sample_freq = raw_eeg.info["sfreq"]
            block_frequencies = psd.psd_welch(raw_eeg, low_freq, high_freq, raw_eeg.times[eeg_start], raw_eeg.times[eeg_end], n_fft = round(sample_freq / step))
            picture_block_data = (block_frequencies, single_eval)
            loaded_data.append(picture_block_data)
            # raw_eeg[channel's, sample's][tuple_elem(0=samples,1=times)]
            block_eeg = np.array(raw_eeg[0:-1, eeg_start:eeg_end + 1][0])
            block_data = (block_eeg, single_eval)
            loaded_data.append(block_data)
    return loaded_data
