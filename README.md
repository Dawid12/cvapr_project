# Emotion Recognition In Images From EEG Signals Using Deep Neural Networks

The aim of the project is to examinate an effectiveness of processing EEG data using neural network. Given a data from conducted experiments the main purpose is to recognize humna's emotion. For this activity Python has been chosen as a lead technology. 

## Technical requirements

### Technologies used

* [Python3](https://www.python.org/)

### Libraries used

* [Keras](https://keras.io/)
* [MNE](https://www.martinos.org/mne/stable/index.html)

## How to run

First, resolve all the technical requirements listed in section above (download and install on your local machine).

Second, download the database from the provided URL (can be shared upon request) and copy the directory where it was downloaded.

Third, clone the project repository.

```bash
$ git clone https://github.com/Dawid12/cvapr_project.git
```

Fourth, navigate to the project directory.

```bash
$ cd cvapr_project/
```

Fifth, run the application providing the directory of the downloaded database, which contains "enterface06_EMOBRAIN" subdirectory, in place of "db_dir".

```bash
$ python main.py "db_dir"
```

## Background

This project has been developed for *Computer Vision and Pattern Recognition* subject by students from the *Silesian University of Technology*.