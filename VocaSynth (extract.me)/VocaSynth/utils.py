import numpy as np

def prepare_sequences(mfccs, sequence_length):
    """
    Prépare les séquences de données pour l'entraînement.
    """
    X = []
    y = []
    for i in range(len(mfccs[0]) - sequence_length):
        X.append(mfccs[:, i:i + sequence_length].T)  # X a la forme (sequence_length, n_mfcc)
        y.append(mfccs[:, i + sequence_length])      # y a la forme (n_mfcc,)
    return np.array(X), np.array(y)

def generate_voice(model, input_sequence, num_steps=100):
    """
    Génère une nouvelle séquence MFCC à partir du modèle entraîné.
    """
    generated = []
    current_sequence = input_sequence.copy()  # Assurez-vous de ne pas modifier la séquence d'entrée

    for _ in range(num_steps):
        prediction = model.predict(np.expand_dims(current_sequence, axis=0))[0]
        generated.append(prediction)

        # Décaler la séquence actuelle
        current_sequence = np.roll(current_sequence, shift=-1, axis=1)
        # Remplacer le dernier élément par la prédiction
        current_sequence[:, -1] = prediction

    return np.array(generated)
