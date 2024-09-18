import librosa

def load_audio(file_path):
    """
    Charger un fichier audio.
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extraire les coefficients MFCC d'un fichier audio.
    """
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
