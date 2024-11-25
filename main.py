from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Initialize FastAPI app
app = FastAPI()

# Load the trained Keras model
model = tf.keras.models.load_model("beat_similarity_filter_model.h5")

# Load RoBERTa tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
embedding_model = AutoModel.from_pretrained("roberta-base").to(device)

# Define Pydantic model for input validation
class InputData(BaseModel):
    petition_body: str
    journalist_articles: List[dict]  # Each article has 'title', 'abstract', 'content'

# Helper function to create embeddings
def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Helper function to create beat embeddings
def create_beat_embedding(articles):
    title_weight = 0.3
    abstract_weight = 0.1
    content_weight = 0.6

    article_embeddings = []
    for article in articles:
        title_emb = embed_text(article.get("title", "")) * title_weight
        abstract_emb = embed_text(article.get("abstract", "")) * abstract_weight
        content_emb = embed_text(article.get("content", "")) * content_weight
        combined_emb = title_emb + abstract_emb + content_emb
        article_embeddings.append(combined_emb)

    if article_embeddings:
        return np.mean(article_embeddings, axis=0)  # Average of all embeddings
    return None  # Return None if no valid articles

@app.post("/predict")
def predict(data: InputData):
    # Step 1: Embed petition
    petition_embedding = embed_text(data.petition_body)

    # Step 2: Embed journalist articles
    journalist_embedding = create_beat_embedding(data.journalist_articles)
    if journalist_embedding is None:
        raise HTTPException(status_code=400, detail="Invalid journalist articles provided.")

    # Step 3: Compute cosine similarity
    similarity = np.dot(petition_embedding, journalist_embedding) / (
        np.linalg.norm(petition_embedding) * np.linalg.norm(journalist_embedding)
    )

    # Step 4: Combine inputs for the model
    combined_input = np.concatenate([petition_embedding, journalist_embedding, [similarity]])
    combined_input = np.expand_dims(combined_input, axis=0)  # Add batch dimension

    # Step 5: Make prediction
    prediction = model.predict(combined_input)[0][0]

    # Step 6: Return result
    return {"prediction": float(prediction)}

