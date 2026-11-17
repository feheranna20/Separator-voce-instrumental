import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

def generate_karaoke_and_acapella(input_path, output_instrumental, output_acapella):
    print(f"Se incarca fisierul original: {input_path} ... ")

    y, sr = librosa.load(input_path, sr=None, mono=False)

    if y.ndim == 1 or y.shape[0] < 2:
        raise ValueError("Fisierul audio trebuie sa fie STEREO!")

    left_channel = y[0]
    right_channel = y[1]

    #Generarea fisierului instrumental (KARAOKE)

    print("Se proceseaza versiunea instrumental ...")

    instrumental_raw = left_channel - right_channel
    y_harmonic_i, y_percussive_i = librosa.effects.hpss(instrumental_raw, margin=1.5)

    sos_hp_inst = butter(N=2, Wn=70.0, btype='hightpass', fs=sr, output='sos')
    instrumental_filtered = sosfilt(sos_hp_inst, y_harmonic_i) + y_percussive_i

    max_inst = np.max(np.abs(instrumental_filtered))
    if max_inst > 0:
        instrumental_filtered = instrumental_filtered / max_inst

    sf.write(output_instrumental, instrumental_filtered, sr)
    print(f"Instrumentalul a fost salvat cu succes in fisierul: {output_instrumental}")

if __name__ == "__main__":
    input_file = "Iris.mp3"
    out_inst = "instrumental_karaoke.wav"

    try:
        generate_karaoke_and_acapella(input_file, out_inst)
        print("Ambele fisiere au fost create cu succes!")
    except Exception as e:
        print(f"A aparut o eroare: {e}")