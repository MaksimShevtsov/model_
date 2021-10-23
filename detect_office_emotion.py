from tensorflow import keras
from keras.models import model_from_json
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import emot as e

# Read in the JSON file
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# Load the model from JSON
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('saved_models/Emotion_Voice_Detection_Model.h5')
print('Loaded model from disk')
# loaded_model.summary()
o = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=o, metrics=['accuracy'])


def readAudioFiles(d, dur, sample_rate):
    if d is None:
        d = 'dir'

    df = pd.DataFrame(columns=['feature'])
    file_names = []
    i = 0
    for audiofile in os.listdir(d):
        if audiofile.endswith('.wav'):
            # Load file using librosa
            print(audiofile, 'loaded')
            file_names.append(audiofile)
            X, sr = librosa.load(os.path.join(d, audiofile), res_type='kaiser_fast', duration=dur, sr=sample_rate,
                                 offset=0.5)
            sr = np.array(sr)
            # Extract the MFCCS
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sr,
                                                 n_mfcc=13),
                            axis=0)
            feature = mfccs
            # Add to data frame
            df.loc[i] = [feature]
            i += 1
    df = pd.DataFrame(df['feature'].values.tolist())
    df = shuffle(df)
    df = df.fillna(0)
    return df, file_names


audio_features, file_names = readAudioFiles(d='the-office-audio-clips', dur=2.5, sample_rate=44100)
audio_features_cnn = np.expand_dims(audio_features, axis=2)


def sumProbs(preds):
    file = []
    for i in range(preds.shape[1]):
        temp = []
        p_angry = preds[i][0] + preds[i][5]
        p_calm = preds[i][1] + preds[i][6]
        p_fearful = preds[i][2] + preds[i][7]
        p_happy = preds[i][3] + preds[i][8]
        p_sad = preds[i][4] + preds[i][9]
        temp.append(p_angry)
        temp.append(p_calm)
        temp.append(p_fearful)
        temp.append(p_happy)
        temp.append(p_sad)
        file.append(temp)
    return np.array(file)


new_preds = sumProbs(preds)


def inverseTransform(preds, emotion_dict):
    decoded = []
    preds = preds.tolist()
    for i in range(len(preds)):
        key = preds[i]
        filename = file_names[i]
        val = emotion_dict[key]
        print('file name:', filename, '/', 'CNN prediction:', key, '/', 'predicted emotion:', val)
        decoded.append(val)
    return filename, key, val


arg_max = new_preds.argmax(axis = 1)
print(arg_max)

emotions = e.emotions

pred_emo = inverseTransform(arg_max, emotions)