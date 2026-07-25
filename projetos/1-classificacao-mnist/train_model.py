import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

#  1. Carregar o dataset MNIST via tf.keras.datasets.mnist <--------------------------------
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#   3. Separar um conjunto de validação (ex: validation_split ou split manual) <--------------------------------
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25)

#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1) <--------------------------------
# Imagens para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Ajuste do shape
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)

#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax) <--------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    # Bloco Convolucional 1
    # Extrai as características
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    # Normaliza os dados
    BatchNormalization(),
    # Resume a informação dividindo por 2
    MaxPooling2D(pool_size=(2, 2)),

    # Bloco Convolucional 2
    # Filter de 64 até 128 para compensar a perda do pooling
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloco Convolucional 3 
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Converte Mapa de características em um vetor
    Flatten(),
    
    # Camada densa com 128 neurônios 
    Dense(128, activation='relu'),

    # Dropout de 50% dos neurônios
    Dropout(0.5),

    # Camada densa de saída com 10 (um para cada dígito)
    Dense(10, activation='softmax')
])

# Resumo do modelo
model.summary()

#   5. Treinar com EarlyStopping monitorando a perda de validação <--------------------------------
from tensorflow.keras.callbacks import EarlyStopping
# Compilando o modelo
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',        # Monitorar a perda no conjunto de validação
    patience=3,                # "Paciência", Quantas épocas esperar sem melhora antes de parar
    restore_best_weights=True  # Salvar os melhores pesos encontrados
)

# Treino do modelo
history = model.fit(
    x_train, y_train,
    epochs=15,                       
    validation_data=(x_val, y_val), 
    callbacks=[early_stopping]       # Parada antecipada
)

#   6. Exibir a acurácia de validação final no terminal <--------------------------------

# Avaliando a CNN treinada
score = model.evaluate(x_val, y_val)

print( '\nPerda:{:.3f}\nAcurácia:{}'.format( score[0], score[1] ) )

#   7. Salvar o modelo treinado como "model.h5" <--------------------------------
model.save('model.h5')