#!/usr/bin/env python3

import sounddevice as sd
import soundfile as sf
import pandas as pd
import pickle
import time

# internal support
from preprocess import preprocess_file, listdir
from project_directories import dataset_dir, pickle_dir

#  sample rate of recording (match sample rate of data)
samplerate = 22050  # [Hz]
# duration of recording (match sample rate of data)
duration = 1        # [s]
# file name to save recording
filename = 'output.wav'


def predict_custom_input(scaler, clf):
    # sleep for 0.1 seconds (adjust reaction time)
    time.sleep(0.1)

    #  record for {duration} seconds at sample rate {samplerate}
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)

    #  write the recording to audio file
    sf.write(filename, mydata, samplerate)

    # play back recording
    print("Here is what I heard...")
    data, fs = sf.read(filename)
    sd.play(data, fs)
    time.sleep(1)

    #  preprocess the data the same way done for the dataset
    X = pd.DataFrame([preprocess_file(filename)[0].flatten()])

    # normalize the data
    X = X.to_numpy()
    X = scaler.transform(X)

    # predict the word associated with this audio file
    predicted = clf.predict(X)

    # report
    print(f"I think you said \'{predicted[0]}\'")


def custom_input(labels, scaler, clf, loop):

    if loop == False:
        # give instructions
        print(f"Say one of the following words: {labels}")
        print("You will be recorded for 1 second, so be quick!")

        # a 5 seconds timer
        t = 5
        while t:
            print("Recording will start in", t, "seconds ", end="\r")
            time.sleep(1)
            t -= 1
        print('\nRecording started!!')

        # predict
        predict_custom_input(scaler, clf)
    else:
        # wait for input
        while input(
            "\nPress ENTER to start recording (you will be recorded for 1 second, so be quick!) " +
                "or Q to exit the program \n" + f"Labels trained: {labels}") == '':
            # predict
            predict_custom_input(scaler, clf)


# main function
if __name__ == '__main__':

    # input labels (given by the subdirectories name)
    input_labels = pd.read_pickle(pickle_dir + 'labels.pkl').drop_duplicates().to_numpy().flatten()

    # load the scaler from the pickle file
    scaler = pickle.load(open(pickle_dir + 'scaler.pkl', 'rb'))

    # load the trained model from pickle file
    clf = pickle.load(open(pickle_dir + 'clf_weights.pkl', 'rb'))

    custom_input(input_labels, scaler, clf, loop=True)
