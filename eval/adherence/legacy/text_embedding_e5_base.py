import torch.nn.functional as F

import torch
import pickle
import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "intfloat/e5-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=".")
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

def embed(docs: list[str]) -> list[list[float]]:
    """
    Embed a list of text to vectors
    """
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


struct_data = pd.read_csv('data/raw_datasets/questions.csv')

# Sample 50 excercises from each subject as train data
sample_subjects_train = struct_data.groupby(by="subject").apply(lambda x: x.sample(n=50)).reset_index(drop=True)
sample_subjects_train.to_csv("data/sample_subjects_train.csv")

# Sample 10 excercises from each subject as train data, different from train data 
test_subjects = struct_data[~struct_data.index.isin(sample_subjects_train.index)]
sample_subjects_test = test_subjects.groupby(by="subject").apply(lambda x: x.sample(n=10)).reset_index(drop=True)
sample_subjects_test.to_csv("data/sample_subjects_test.csv")


samples = sample_subjects_train["content"].to_list()

# Embed train text to vectors in batch size of 10
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

# Save vectors to file
with open('data/sample_vectors_train.pkl', 'wb') as f:
    pickle.dump(vectors, f)


samples = sample_subjects_test["content"].to_list()

# Embed test text to vectors in batch size of 10
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

# Save vectors to file
with open('data/sample_vectors_test.pkl', 'wb') as f:
    pickle.dump(vectors, f)







