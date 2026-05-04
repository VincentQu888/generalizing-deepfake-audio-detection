import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def stft(audio_path: str, window_size: int = 1024, hop_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    compute the short-time fourier transform (stft) of an audio file

    audio_path: path to the input audio file
    window_size: size of fft window
    hop_length: number of samples between successive frames

    returns:
    magnitude_db: magnitudes of each stft bin in decibels
    phase: phase information
    """

    # load audio
    audio, sr = librosa.load(audio_path, sr=None)

    # compute stft
    stft_result = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)

    # compute magnitude and phase
    magnitude = np.abs(stft_result)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    phase = np.angle(stft_result)

    # plot results
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(magnitude_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Spectrogram (dB)")
    # plt.tight_layout()
    # plt.show()

    return magnitude_db, phase