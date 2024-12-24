import torch
import numpy as np
import pickle
import pandas as pd
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)

texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
def embed(docs: list[str]) -> list[list[float]]:
    embeddings = model.encode(docs, task="classification")
    return torch.from_numpy(embeddings).detach().cpu()

sample_subjects = pd.read_csv("data/sample_subjects.csv")

samples = sample_subjects["content"].to_list()

vectors = None
step = 5
for i in range(0, len(samples), step):
    print(f"Processing {i}-{i+step}")
    if vectors is not None:
        vectors = np.concatenate([vectors, embed(samples[i:i+step])], axis=0)
        pass
    else:
        vectors = embed(samples[i:i+step])
        pass
    # vectors = embed(samples[i:i+step])

with open('data/sample_vectors_jina.pkl', 'wb') as f:
    pickle.dump(vectors, f)