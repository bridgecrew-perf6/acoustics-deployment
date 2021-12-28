import numpy as np
import cv2
import librosa

melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}

pcen_parameters = {
    "gain": 0.98,
    "bias": 2,
    "power": 0.5,
    "time_constant": 0.4,
    "eps": 0.000001
}
def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def get_features(audio, TARGET_SR):           
    melspec = librosa.feature.melspectrogram(audio, sr=TARGET_SR, **melspectrogram_parameters)
    pcen = librosa.pcen(melspec, sr=TARGET_SR, **pcen_parameters)
    clean_mel = librosa.power_to_db(melspec ** 1.5)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    norm_melspec = normalize_melspec(melspec)
    norm_pcen = normalize_melspec(pcen)
    norm_clean_mel = normalize_melspec(clean_mel)
    image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width * 224 / height), 224))
    image = np.moveaxis(image, 2, 0)
    image = (image / 255.0).astype(np.float32)
    return image
