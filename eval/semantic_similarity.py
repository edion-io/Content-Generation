import re
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SemanticSimilarity:
    """
    A class for computing semantic similarity between texts using multiple transformer-based models.

    Supports Sentence-Transformers and BERT-style models with configurable pooling strategies,
    model weights, and power transforms to combine similarity scores.
    """

    def __init__(self,
                 models=[
                     "intfloat/multilingual-e5-large",
                     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 ],
                 modes=None,
                 weights=None,
                 power_transform=None,
                 is_sentence_transformer=None,
                 device=None,
                 **kwargs
                 ) -> None:
        """
        Initialize the SemanticSimilarity object with one or more transformer models.

        Args:
            models (list): List of model names to load.
            modes (list or str): Pooling modes ("CLS" or "token_mean") for each model.
            weights (list): Weights to combine similarity scores from multiple models.
            power_transform (list): Exponents to apply to individual similarity scores before weighting.
            is_sentence_transformer (list): Whether each model is a Sentence-Transformer model.
            device (str or torch.device): Device to load models onto (e.g., "cuda" or "cpu").
            **kwargs: Additional unused keyword arguments.
        """
        self.models = {}
        self.tokenizers = {}

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if is_sentence_transformer is None:
            is_sentence_transformer = [m.lower().find("bert") == -1 for m in models]

        for model, is_st in zip(models, is_sentence_transformer):
            print(f"Loading {model}...")
            if is_st:
                self.tokenizers[model] = AutoTokenizer.from_pretrained(model)
                self.models[model] = AutoModel.from_pretrained(model)
                self.models[model].to(self.device)
            else:
                self.models[model] = BertModel.from_pretrained(model)
                self.tokenizers[model] = BertTokenizer.from_pretrained(model)
        self.model_is_st = dict(zip(models, is_sentence_transformer))

        if weights is None:
            weights = [1.0] * len(models)
        weights = np.array(weights) / sum(weights)
        self.weights = dict(zip(models, weights))

        if power_transform is None:
            power_transform = [1.0] * len(models)
        self.power_transform = dict(zip(models, power_transform))

        if modes is None:
            modes = "CLS"
        if isinstance(modes, str):
            modes = [modes] * len(models)
        self.modes = {model: mode.lower() for model, mode in zip(models, modes)}

    def _cosine_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two 1D embeddings.

        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        if embedding1.ndim != 1 or embedding2.ndim != 1:
            raise ValueError(f"Both embeddings must be 1D (actual shapes {embedding1.shape} and {embedding2.shape})")
        return 1 - cosine(embedding1, embedding2)

    def _chunk_text(self, text, max_tokens, tokenizer):
        """
        Break a long text into chunks that fit within the model's token limit.

        Args:
            text (str): Input text.
            max_tokens (int): Maximum tokens per chunk.
            tokenizer: The tokenizer for the model.

        Returns:
            list: A list of tokenized chunks.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling over token embeddings with attention mask.

        Args:
            model_output (tuple): Output from transformer model.
            attention_mask (torch.Tensor): Mask to ignore padding tokens.

        Returns:
            torch.Tensor: Mean pooled sentence embedding.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def individual_embedding(self,
                             model,
                             text,
                             clean_latex=True,
                             max_length=512
                             ) -> np.ndarray:
        """
        Generate an embedding for a single piece of text using the specified model.

        Args:
            model (str): Model name to use for embedding.
            text (str): Input text.
            clean_latex (bool): Whether to clean LaTeX-like formatting from the text.
            max_length (int): Max length for BERT-style token chunking.

        Returns:
            np.ndarray: Normalized text embedding.
        """
        if self.model_is_st[model]:
            if clean_latex:
                text = self.clean_latex(text)
            encoded_input = self.tokenizers[model](text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = self.models[model](**encoded_input)
            if self.modes[model] == "token_mean":
                doc_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            elif self.modes[model] == "cls":
                doc_embedding = model_output[0][:, 0, :]
            else:
                raise ValueError(f"Unknown mode {self.modes[model]}")
            doc_embedding = F.normalize(doc_embedding, p=2, dim=1).cpu().squeeze().numpy()
        else:
            chunks = self._chunk_text(text, max_length - 2, self.tokenizers[model])
            embeddings = []
            for chunk in chunks:
                input_ids = torch.tensor([self.tokenizers[model].build_inputs_with_special_tokens(chunk)])
                with torch.no_grad():
                    outputs = self.models[model](input_ids)
                if self.modes[model] == "token_mean":
                    embeddings.append(self._mean_pooling(outputs, input_ids.ne(self.tokenizers[model].pad_token_id)).numpy())
                elif self.modes[model] == "cls":
                    embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
            doc_embedding = np.mean(embeddings, axis=0).squeeze()
        return doc_embedding

    def embeddings_dict(self, text, clean_latex=True):
        """
        Generate embeddings for a text using all configured models.

        Args:
            text (str): Input text.

        Returns:
            dict: Dictionary of model names and their respective embeddings.
        """
        embeddings = dict()
        for model in self.models.keys():
            embeddings[model] = self.individual_embedding(model, text, clean_latex)
        return embeddings

    def clean_latex(self, text):
        """
        Clean LaTeX-specific artifacts from text.

        Args:
            text (str): Raw text possibly containing LaTeX formatting.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r"\\[\w\*]*", "", text)
        text = re.sub(r"\{.*?\}", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        return text.strip()

    def get_similarity(self,
                       in1: str | dict[str, np.ndarray],
                       in2: str | dict[str, np.ndarray],
                       power_transform=True
                       ):
        """
        Compute the weighted semantic similarity between two texts or embeddings.

        Args:
            in1 (str or dict): First input (text or precomputed embeddings).
            in2 (str or dict): Second input (text or precomputed embeddings).
            power_transform (bool): Whether to apply power transforms to similarity scores.

        Returns:
            float: Weighted similarity score between 0 and 1.
        """
        if isinstance(in1, str):
            in1 = self.clean_latex(in1)
        if isinstance(in2, str):
            in2 = self.clean_latex(in2)
        similarities = []
        for model in self.models.keys():
            if isinstance(in1, str):
                out1 = self.individual_embedding(model, in1)
            else:
                out1 = in1[model]
            if isinstance(in2, str):
                out2 = self.individual_embedding(model, in2)
            else:
                out2 = in2[model]
            similarity = self._cosine_similarity(out1, out2)
            if power_transform:
                similarity = similarity ** self.power_transform[model]
            similarity *= self.weights[model]
            similarities.append(similarity)

        similarities = np.array(similarities)
        assert similarities.sum() <= 1
        return similarities.sum()


if __name__ == "__main__":
    text1 = "Write the following fractions in order from smallest to largest: 3/4, 2/3, 5/6. Explain how you determined the order."
    text2 = "Order these fractions from smallest to largest: 2/3, 3/4, 5/6. Explain your method and reasoning."

    print(SemanticSimilarity().get_similarity(text1, text2))