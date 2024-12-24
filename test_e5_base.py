import torch.nn.functional as F

import torch
import re
import pickle
import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "intfloat/e5-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

def embed(docs: list[str]) -> list[list[float]]:
    docs = [f"passage: {d}" for d in docs]
    # tokenize
    tokens = tokenizer(
        docs, padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        # process with model for token-level embeddings
        out = model(**tokens)
        # mask padding tokens
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        # create mean pooled embeddings
        doc_embeds = last_hidden.sum(dim=1) / \
            tokens["attention_mask"].sum(dim=1)[..., None]
    return doc_embeds.detach().cpu()

sample_subjects = pd.read_csv("data/sample_subjects_train.csv")

samples = sample_subjects["content"].to_list()

vectors = None
step = 10
for i in range(0, len(samples), step):
    print(f"Processing {i}-{i+step}")
    if vectors is not None:
        vectors = torch.cat([vectors, embed(samples[i:i+step])], dim=0)
        pass
    else:
        vectors = embed(samples[i:i+step])
        pass
    # vectors = embed(samples[i:i+step])

with open('data/sample_vectors_train.pkl', 'wb') as f:
    pickle.dump(vectors, f)



sample_subjects = pd.read_csv("data/sample_subjects_test.csv")

samples = sample_subjects["content"].to_list()

vectors = None
step = 10
for i in range(0, len(samples), step):
    print(f"Processing {i}-{i+step}")
    if vectors is not None:
        vectors = torch.cat([vectors, embed(samples[i:i+step])], dim=0)
        pass
    else:
        vectors = embed(samples[i:i+step])
        pass
    # vectors = embed(samples[i:i+step])

with open('data/sample_vectors_test.pkl', 'wb') as f:
    pickle.dump(vectors, f)







