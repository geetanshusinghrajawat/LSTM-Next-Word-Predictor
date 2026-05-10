# 🔤 Next Word Prediction (LSTM)

An LSTM-based language model that predicts the next word(s) from a given seed text — trained on real news headlines and deployed as an interactive Streamlit app.

## 🚀 Live Demo
[**Try it here →**](https://lstm-next-word-predictor-by-geetanshu.streamlit.app/)

## 📌 Overview

Type any starting text and the app generates the most likely next words using a trained LSTM model. The model learns sequential word patterns from 20,000 news headlines and generates contextually relevant predictions.

## 🛠️ How It Works

1. **Data** — 20,000 news headlines sampled from the News Category Dataset
2. **Tokenization** — Keras Tokenizer builds a vocabulary of 17,000+ unique tokens
3. **Sequence Generation** — N-gram sequences created from each sentence; padded to max length (185)
4. **Model** — Embedding (vocab × 128) → LSTM (256 units) → Dense (vocab size, Softmax)
5. **Training** — 10 epochs, Adam optimizer, sparse categorical crossentropy loss; 80/20 train-test split
6. **Inference** — Iteratively predicts and appends next words to generate fluent continuations
7. **Deployment** — Streamlit app with adjustable word count slider

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | Embedding + LSTM + Dense |
| Preprocessing | Keras Tokenizer, pad_sequences |
| Deployment | Streamlit |

## 📁 Project Structure

```
├── nextword_lstm.ipynb    # Model training notebook
├── app.py                 # Streamlit web app
├── nextword_model.h5      # Saved Keras model
└── tokenizer.pickle       # Serialized tokenizer
```

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/geetanshusinghrajawat/your-repo-name
cd your-repo-name

# Install dependencies
pip install streamlit tensorflow numpy pandas

# Run the app
streamlit run app.py
```

## 📊 Dataset

[News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) — 200,000+ news headlines from HuffPost across multiple categories. 20,000 headlines used for training.

## 👤 Author

**Geetanshu Singh Rajawat**
[LinkedIn](https://www.linkedin.com/in/geetanshu-singh-rajawat/) | [GitHub](https://github.com/geetanshusinghrajawat)
