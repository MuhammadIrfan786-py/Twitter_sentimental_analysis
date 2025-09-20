import sys
import os

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from src.models.rnn import RnnSentimentClassifier
from src.utils import load_vocab, load_model, predict_sentiment

app = FastAPI()

# configurable parameters
MODEL_PATH = 'artifacts/rnn_sentiment_model.pth'
VOCAB_PATH = 'artifacts/vocab.pkl'
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 2
NUM_LAYERS = 2

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load vocab and model
vocab = load_vocab(VOCAB_PATH)
print(f"Loaded Vocab of size: {len(vocab)}")

model = RnnSentimentClassifier(
    vocab_size=len(vocab),
     embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes= NUM_CLASSES,
        num_layers=NUM_LAYERS,
)

model = load_model(model, model_path=MODEL_PATH, device=device)

print(f"Model loaded and ready for inference. {model}")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def read_item(query: str):
    sentiment, confidence = predict_sentiment(model, vocab, query, device)
    return {"sentiment": sentiment, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)