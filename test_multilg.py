import torch.nn.functional as F

import torch
import re
import pickle
import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct', cache_dir=".")
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', device_map="auto")
model.eval()


def embed(docs: list[str]) -> list[list[float]]:
    # Tokenize the input texts
    batch_dict = tokenizer(docs, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings.detach().cpu()


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





