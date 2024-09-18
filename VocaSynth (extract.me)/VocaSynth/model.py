import tensorflow as tf

def create_rnn_model(input_shape):
    """
    Crée et compile un modèle RNN.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dense(input_shape[1])  # Nombre de sorties égal à la dimension des MFCCs
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
