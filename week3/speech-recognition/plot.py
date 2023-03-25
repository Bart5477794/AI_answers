#!/usr/bin/env python3
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import glob

# internal support
from utils import filename_no_ext
from preprocess import preprocess_file
from project_directories import raw_data_dir, plot_dir


def plot_spectrogram(S_dB, sr, label='', show=False):
    # plot spectrogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f"Mel-frequency spectrogram {label}")
    plt.xlim([0, 1])
    if show:
        plt.show()
    return fig


def plot_sample_spectrogram(label, sample=0, show=False):
    # if it does not exist yet, create a directory for output files
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # select one file per label
    filename = glob.glob(raw_data_dir + label + "/*.wav")[sample]

    # preprocess spectrogram for one file
    [S_dB, sr] = preprocess_file(filename)
    figname = label + '_' + filename_no_ext(filename) + '_mel' + '.png'

    # visualize spectrogram
    plot_spectrogram(S_dB, sr, label='(' + label + ')', show=show).savefig(plot_dir + '/' + figname)


def plot_signal(S, label='', show=False):
    # map time samples to time (final time is 1s)
    t = (1 + np.arange(len(S))) / len(S)

    # plot signal
    fig, ax = plt.subplots()
    plt.plot(t, S)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 1])

    ax.set_title(f"Signal {label}")
    if show:
        plt.show()
    return fig


def plot_sample_signal(label, sample=0, show=False):
    # if it does not exist yet, create a directory for output files
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # select one file per label
    filename = glob.glob(raw_data_dir + label + "/*.wav")[sample]

    # load raw data
    y, _ = librosa.load(filename)
    figname = label + '_' + filename_no_ext(filename) + '.png'

    # visualize signal
    plot_signal(y, label='(' + label + ')', show=show).savefig(plot_dir + '/' + figname)


# main function
if __name__ == '__main__':

    """
    TODO:
    Part 0, Step 0: Visualize additional samples signals
    """
    # plot the signal for sample number 0 of the data labelled with "one"
    plot_sample_signal("one", sample=0)
    plot_sample_signal("one", sample=1)
    plot_sample_signal("one", sample=2)

    plot_sample_spectrogram("one", sample=0)
    plot_sample_spectrogram("one", sample=1)
    plot_sample_spectrogram("one", sample=2)
    """
    TODO:
    Part 0, Step 1: Visualize the Mel spectrograms for selected samples
    """
