import os

# Définir la variable d'environnement avant d'importer TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from audio_processing import load_audio, extract_mfcc
from model import create_rnn_model
from utils import prepare_sequences, generate_voice
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import soundfile as sf

def mfcc_to_audio(mfccs, sr):
    """
    Convertit les MFCCs générés en audio.
    """
    # Reconstruction du spectrogramme à partir des MFCCs
    spectrogram = librosa.feature.inverse.mfcc_to_audio(mfccs.T)
    return spectrogram

# Charger l'audio et extraire les MFCC
file_path = 'Sample_Test_Voix1.wav'
print("Chargement de l'audio...")
y, sr = load_audio(file_path)
mfccs = extract_mfcc(y, sr)
print(f"Audio chargé. Forme des MFCCs : {mfccs.shape}")

# Préparer les séquences pour l'entraînement
sequence_length = 30
print("Préparation des séquences...")
X, y = prepare_sequences(mfccs, sequence_length)
print(f"Forme des séquences X : {X.shape}")
print(f"Forme des étiquettes y : {y.shape}")

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer et entraîner le modèle
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_mfcc)
print("Création du modèle...")
model = create_rnn_model(input_shape)
model.summary()  # Afficher l'architecture du modèle
print("Entraînement du modèle...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Générer la voix à partir du modèle
print("Génération de la voix...")
input_sequence = X_test[0]
generated_mfcc = generate_voice(model, input_sequence)
print(f"Forme des MFCC générés : {generated_mfcc.shape}")

# Sauvegarder le fichier audio généré
print("Sauvegarde du fichier audio...")
generated_audio = mfcc_to_audio(generated_mfcc, sr)
sf.write('generated_voice.wav', generated_audio, sr)
print("Voix générée et sauvegardée sous 'generated_voice.wav'")
