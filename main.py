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

    #Generarea fisierului voce (ACAPELLA)

    print("Se proceseaza versiunea acapella ...")

    center_channel = 0.5 * (left_channel + right_channel)

    sos_vocal_hp = butter(N=3, Wn=120.0, btype='highpass', fs=sr, output='sos')
    center_clean = sosfilt(sos_vocal_hp, center_channel)


    D = librosa.stft(center_clean, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(D)

    background = librosa.decompose.nn_filter(magnitude, aggregate=np.median, metric='cosine', axis=1)

    epsilon=1e-8
    mask=(magnitude ** 2) / ((magnitude ** 2) + (1.6 * background ** 2) + epsilon)

    clean_magnitude=magnitude * mask

    D_vocal = clean_magnitude * phase
    acapella_raw = librosa.istft(D_vocal, hop_length=512, length=len(center_clean))

    sos_bp = butter(N=4, Wn=[150.0, 3800.0], btype='bandpass', fs=sr, output='sos')
    acapella_filtered = sosfilt(sos_bp, acapella_raw)

    max_acap = np.max(np.abs(acapella_filtered))
    if max_acap > 0:
        acapella_filtered = acapella_filtered / max_acap
        
    sf.write(output_acapella, acapella_filtered, sr)
    print(f"Partea acapella a fost salvata cu succes in fisierul: {output_acapella}")


if __name__ == "__main__":
    input_file = "Iris.mp3"
    out_inst = "instrumental_karaoke.wav"
    out_acap = "acapella_voce.wav"

    try:
        generate_karaoke_and_acapella(input_file, out_inst, out_acap)
        print("Ambele fisiere au fost create cu succes!")
    except Exception as e:
        print(f"A aparut o eroare: {e}")