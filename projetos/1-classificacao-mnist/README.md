# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Felipe Vieira de Oliveira

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura da CNN foi construída utilizando 3 blocos convolucionais, onde cada bloco é composto por convolução, BatchNormalization para estabilização do aprendizado, e MaxPooling2D para redução de dimensionalidade. Antes da camada de saída temos uma camada de Dropout como técnica para evitar overfitting. A estratégia de validação utilizada foi o early stopping, que aguarda algumas épocas sem melhora do algoritmo para interromper a busca de melhores parâmetros.

Foram utilizados hiperparâmetros como o Dropout(0.5), para desativar 50% dos neurônios durante o treino, o que força a rede a aprender características mais generalistas do dataset

### 2️⃣ Bibliotecas Utilizadas

Foram utilizadas as bibliotecas recomendadas do projeto, que foram o tensorflow (v2.12), para mexer com CNN, numpy, para manipulação de arrays. Também foi utilizado scikit-learn, para fazer a divisão dos dados, criando o conjunto de validação.

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica 'Dynamic Range Quantization', que converte os pesos do modelo de ponto flutuante float para números inteiros int. Isso reduz o tamanho do modelo e o uso de memória, com pouca perda de acurácia, como mostra o próximo tópico.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

Para o model.h5 a acurácia foi de aproximadamente 0.9867, com o tamanho do arquivo de 1378KB

Para o modelo quantizado a acurácia foi de aproximadamente 0.9873, com o tamanho do arquivo de 122KB

### 5️⃣ Comentários Adicionais (Opcional)

Observando os resultados obtidos, percebe-se a importância da quantização para dispositivos Edge.
A técnica reduziu grandemente o espaço utilizado, comprimindo o arquivo em mais de 10 vezes, enquanto a acurácia sofreu um leve aumento, não o usual, de 0.9867 para 0.9873. Isso comprova a a importância e necessidade dessas otimizações para permitir que esses modelos rodem de forma eficiente em dispositivos carentes de armazenamento e processamento.

Em relação a dificuldades no desenvolvimento desse projeto, tive um problema técnico significativo que foi o gerenciamento de compatibilidade entre o ambiente local de treinamento e o Action de avaliação automatica do GitHub. Como o TensorFlow adotou o Keras 3 como padrão nas versões mais recentes, o arquivo .h5 gerado localmente passou a adotar uma estrutura incompatível com ambientes que ainda dependem do motor do Keras 2.

Para contornar essa limitação precisei colocar a variável de ambiente os.environ['TF_USE_LEGACY_KERAS'] = '1' diretamente no topo dos códigos, isolando a execução local. Isso forçou o TensorFlow a salvar os pesos e a arquitetura no formato legado. Esse desafio foi uma alusão a uma realidade na área de Edge IA e embarcados, que é o esforço para garantir que um modelo treinano em um ambiente moderno seja suportado no dispositivo final.

### 6️⃣ Exemplo de Inferência

Fazendo a prova real com 5 amostra, tivemos:

Amostra 1: predito=7 | real=7

Amostra 2: predito=2 | real=2

Amostra 3: predito=1 | real=1

Amostra 4: predito=0 | real=0

Amostra 5: predito=4 | real=4

Em 5 amostras não foram identificados erros, mas espera-se que ocorra erro em alguma amostra futura, considerando a acurácia dos dois modelos
