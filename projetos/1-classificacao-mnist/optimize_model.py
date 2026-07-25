import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

#   1. Carregar o modelo treinado em "model.h5" <---------------------
model = tf.keras.models.load_model('model.h5')
# model = tf.keras.models.load_model('model.h5')

#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter <---------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization, 
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT]) <---------------------
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converter
tflite_model = converter.convert()

#   4. Salvar o resultado como "model.tflite" <---------------------
with open("model.tflite", 'wb') as f:
    f.write(tflite_model)

# -> CALCULAR ACURÁCIA
from keras.datasets import mnist
import numpy as np
# Carregar dados
(_, _), (x_val, y_val) = mnist.load_data()

# Normalizar
x_val = x_val / 255.0

# tflite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# detalhes ta entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

acertos = 0
total_amostras = len(x_val)

# Inferência imagem por imagem nos dados de validação
for i in range(total_amostras):
    
    formato_esperado = input_details[0]['shape']
    input_data = x_val[i].reshape(formato_esperado).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    # Pega o resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Classe previsto pelo modelo
    predicao = np.argmax(output_data)
    
    rotulo_verdadeiro = y_val[i] 
    
    # Check se acertou
    if predicao == rotulo_verdadeiro:
        acertos += 1

# Acurácia
acuracia_tflite = acertos / total_amostras
print(f"Acurácia de validação do modelo TFLite otimizado: {acuracia_tflite:.4f}")