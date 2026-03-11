# QuoteGenie: An AI-Powered Text Generation Model

## Overview
This project delves into the fascinating world of text generation using Long Short-Term Memory (LSTM) neural networks, a powerful type of recurrent neural network. Developed with TensorFlow and Keras, the core objective is to train a model on a diverse dataset of quotes, enabling it to learn the nuances of language and then generate new, coherent, and contextually relevant text based on a user-provided seed phrase.

The inspiration behind this project is to explore the creative capabilities of AI in generating human-like text, potentially for inspirational content, creative writing assistance, or simply to marvel at the patterns AI can discern in language.

## Features

-   **Data Acquisition & Preparation**: Reads quote data from a CSV file (`qoute_dataset.csv`).
-   **Robust Text Preprocessing**: Transforms raw text by converting all characters to lowercase and meticulously removing punctuation, ensuring a clean and consistent input for the model.
-   **Advanced Tokenization**: Utilizes Keras's `Tokenizer` to convert text into numerical sequences, building a vocabulary of `8978` unique words.
-   **Sequence Formulation**: Constructs input-output pairs by creating sequences where each input is a prefix of a sentence and the output is the next word, essential for training a sequential model.
-   **Dynamic Padding**: Employs `pad_sequences` to ensure all input sequences have a uniform length (`max_len = 745`), a critical step for batch processing in neural networks.
-   **LSTM Neural Network Architecture**: Implements a `Sequential` model comprising:
    -   An `Embedding` layer (`output_dim=50`) to convert word indices into dense vectors.
    -   A `SimpleRNN` or `LSTM` layer (`units=128`) to capture temporal dependencies within sequences.
    -   A `Dense` output layer with `softmax` activation (`units=vocab_size`) for multi-class classification (predicting the next word).
-   **Model Training**: Trains the LSTM model using `categorical_crossentropy` loss and the `adam` optimizer over `100` epochs with a `batch_size` of `128`.
-   **Next Word Prediction**: A `predictor` function efficiently takes a seed text and the trained model to predict the most probable next word.
-   **Iterative Text Generation**: A `generate_text` function extends a given seed phrase by iteratively predicting and appending subsequent words, allowing for the creation of longer, cohesive sentences or paragraphs.
-   **Model Persistence**: Saves the trained `lstm_model` in H5 format (`lstm_model.h5`) and the `tokenizer` and `max_len` using `pickle` (`tokenizer.pkl`, `max_len.pkl`) for easy deployment and future use without requiring retraining.

## Technologies Used

*   **Python 3.x**
*   **TensorFlow/Keras**: For building, training, and evaluating the deep learning model.
*   **Pandas**: Essential for data manipulation and analysis.
*   **NumPy**: For numerical operations, particularly with arrays.
*   **`string` module**: For basic string operations like punctuation removal.
*   **`pickle`**: For serializing Python objects (tokenizer and max_len).

## Getting Started

### Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/22-vaibhav/QuoteGenie.git
    cd QuoteGenie
    ```
2.  Install the required Python packages:
    ```bash
    pip install tensorflow pandas numpy
    ```

### Dataset
The project uses a `qoute_dataset.csv` file, which should be placed in the root directory of the project. This dataset contains the quotes used for training the model.

### Training the Model
Run the provided Jupyter Notebook (or Python script) to load the data, preprocess it, define the LSTM model, and train it. The notebook handles all these steps sequentially.

### Generating Text
After the model is trained and saved, you can use the `predictor` and `generate_text` functions to create new text. Here's a quick example:

```python
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved model and tokenizer
lstm_model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokinizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Recreate index_to_word mapping
word_index = tokinizer.word_index
index_to_word = {}
for word, index in word_index.items():
    index_to_word[index] = word

def predictor(model,tokenizer,text,max_len):
  text = text.lower()
  seq = tokenizer.texts_to_sequences([text])[0]
  seq = pad_sequences([seq], maxlen=max_len, padding='pre')
  pred = model.predict(seq, verbose = 0)
  pred_index = np.argmax(pred)
  return index_to_word[pred_index]

def generate_text(model,tokenizer,seed_text,max_len,n_words):
  for _ in range(n_words):
    next_word = predictor(model,tokenizer,seed_text,max_len)
    if next_word == "": # Handle cases where no next word is predicted
      break
    seed_text += " " + next_word
  return seed_text

# Example usage:
seed = "the world is a"
generated_sentence = generate_text(lstm_model, tokinizer, seed, max_len, 10)
print(generated_sentence)
# Expected output might be something like: 'the world is a beautiful place to live in and it is'
```

## Model Details
- Vocabulary Size: 8978 unique words.
- Embedding Dimension: 50 (Each word is represented by a 50-dimensional vector).
- LSTM Units: 128 (The number of hidden units in the LSTM layer).
- Input Sequence Length: 745 (Maximum length of input sequences after padding).
- Training Epochs: 100
- Batch Size: 128

## Future Enhancements
- Experiment with different model architectures (e.g., GRUs, Transformers).
- Explore larger and more diverse datasets for richer text generation.
- Implement temperature sampling for more varied and creative outputs.