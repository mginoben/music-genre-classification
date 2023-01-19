from keras import models
import librosa
import numpy as np
import wave
import pyaudio
import PySimpleGUI as sg

model = models.load_model('model')

classes = {
    0: 'blues',
    1: 'Classical',
    2: 'Country',
    3: 'Disco',
    4: 'Hiphop',
    5: 'Jazz',
    6: 'Metal',
    7: 'Pop',
    8: 'Reggae',
    9: 'Rock',
}


# Function to extract features
def features_extractor():
    data, sample_rate = librosa.load('test.wav', res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


def predictClass():
    prediction_features = features_extractor()
    prediction_features = prediction_features.reshape(1, -1)

    predict_x = model.predict(prediction_features)
    classes_x = np.argmax(predict_x, axis=1)
    class_index = classes_x[0]

    return classes.get(class_index)


def recordAudio():
    audio = pyaudio.PyAudio()

    RATE = 44100
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    RECORD_SEC = 10
    CHUNK = 1024
    SPEAKERS = audio.get_device_count()
    OUTPUT_FILENAME = 'test.wav'

    # Print indexes of devices
    for i in range(audio.get_device_count()):
        print(audio.get_device_info_by_index(i))

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=1)  # Index of you default recording device

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SEC)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



layout = [
    [sg.Button('Record', size=(25, 2), key='rec')],
    [sg.Button('Exit', size=(25, 2))]
]

margins = (100, 50)

window = sg.Window(title='ARTINTE', layout=layout, margins=margins)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == 'rec':
        recordAudio()
        class_pred = predictClass()
        sg.popup('Predicted Genre is ' + class_pred)

window.close()
