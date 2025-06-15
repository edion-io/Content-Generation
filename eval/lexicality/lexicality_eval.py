"""
Lexicality Evaluation Module

This module provides functions for evaluating the lexical similarity between generated text 
and reference text. It implements various metrics including:

- Flesch Reading Ease (FRE) similarity
- ROUGE scores (ROUGE-1 and ROUGE-S)
- Sentence length and count similarity 
- Punctuation distribution similarity

The main function `lexicality()` computes all metrics and returns a weighted score based on 
predefined weights. The module is designed to evaluate both individual text pairs and 
population-level statistics.

Key Functions:
    - lexicality(): Main function to compute all lexicality metrics
    - flesch_reading_ease(): Calculate Flesch Reading Ease score
    - rouge_scores(): Calculate ROUGE-1 and ROUGE-S F1 scores
    - calculate_population_sentence_metrics(): Calculate average sentence metrics across texts
    - calculate_punct_distribution_similarity(): Compare punctuation distributions
    - calculate_weighted_score(): Compute final weighted score from metrics

The module uses NLTK for text processing and scipy for statistical calculations.
"""

import re
import nltk
from nltk import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
import textstat

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords') 


# nltk.download('punkt')

DEFAULT_WEIGHTS = {
    'fre_similarity': 0.0721,
    'sentence_length_similarity': 0.0012,
    'sentence_count_similarity': 0.0741,
    'rouge1': 0.3252,
    'rougeS': 0.5252,
    'punctuation_similarity': 0.0023
}

def relative_difference(pred, ref):
    return 1 - abs(abs(pred - ref) / max(pred, ref))

# Function to calculate FRE score
def flesch_reading_ease(text):
    return textstat.flesch_reading_ease(text)

# Function to calculate ROUGE-1 and ROUGE-S F1 scores
def rouge_scores(prediction, reference):
    """Calculate ROUGE-1 and ROUGE-S F1 scores between prediction and reference."""
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    # Extract F1 scores for ROUGE-1 and ROUGE-S
    rouge_1_f1 = scores['rouge1'].fmeasure
    rouge_s_f1 = scores['rouge2'].fmeasure 
    return rouge_1_f1, rouge_s_f1

# Function to calculate average sentence length
def average_sentence_length(text):
    """Calculate the average number of words per sentence in the text."""
    # Tokenize the input text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    # Return the average number of words per sentence
    return len(words) / len(sentences)

# Function to calculate average sentence count (based on number of periods)
def average_sentence_count(text):
    """Count the number of sentences in the text based on period count."""
    # Count the number of periods in the text
    sentence_count = text.count('.')
    return sentence_count

def punctuation_distribution(text):
    """
    Calculate the distribution of selected punctuation marks in the text.
    Apostrophes, quotes, and backticks are only counted when they appear
    as standalone punctuation (not inside words).
    """
    # Regex explanation:
    # - [.!?;:,\-—] = standard punctuation
    # - (?<!\w)['"`](?!\w) = quote/backtick not surrounded by letters/numbers (not in a word)
    pattern = r'[.!?;:,\-—]|(?<!\w)[\'"`](?!\w)'

    matches = re.findall(pattern, text)
    punct_counts = Counter(matches)
    total_punct = sum(punct_counts.values()) or 1  # avoid division by 0

    # Define mapping of character to label
    punct_keys = {
        '.': 'period_ratio',
        '!': 'exclamation_ratio',
        '?': 'question_ratio',
        ':': 'colon_ratio',
        ';': 'semi_colon_ratio',
        '-': 'hyphen_ratio',
        '—': 'em_dash_ratio',
        '`': 'backtick_ratio',
        ',': 'comma_ratio',
        '"': 'quote_ratio',
        "'": 'apostrophe_ratio'
    }

    # Calculate ratios
    ratios = {
        name: punct_counts.get(char, 0) / total_punct
        for char, name in punct_keys.items()
    }

    return ratios

# Function to calculate population-level punctuation distribution
def calculate_population_punct_distribution(texts):
    """
    Calculate punctuation distribution across the entire population of texts as one large text
    """
    # Concatenate all texts with spaces to create one large text
    combined_text = " ".join(texts)
    # Calculate distribution on the combined text
    return punctuation_distribution(combined_text)

# Function to calculate similarity between two punctuation distributions
def calculate_punct_distribution_similarity(pred_dist, ref_dist):
    """
    Calculate similarity between prediction and reference punctuation distributions
    using Jensen-Shannon divergence
    """
    pred_array = np.array([pred_dist[k] for k in pred_dist])
    ref_array = np.array([ref_dist[k] for k in ref_dist])
    
    # Calculate JS divergence and convert to similarity score
    js_div = jensenshannon(pred_array, ref_array)
    similarity_score = 1 - js_div  # Convert distance to similarity
    return similarity_score

# Function to calculate population-level sentence metrics
def calculate_population_sentence_metrics(texts):
    """
    Calculate average sentence length and count across the entire population of texts
    """
    total_words = 0
    total_sentences = 0
    
    for text in texts:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        total_words += len(words)
        total_sentences += len(sentences)
        
    
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    avg_sentence_count = total_sentences / len(texts) if texts else 0
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'avg_sentence_count': avg_sentence_count
    }

# Function to calculate similarity between two sentence metric distributions
def calculate_sentence_metrics_similarity(pred_stats, ref_stats):
    """
    Calculate similarity between prediction and reference sentence metric distributions
    """
    length_similarity = relative_difference(pred_stats['avg_sentence_length'], ref_stats['avg_sentence_length'])
    count_similarity = relative_difference(pred_stats['avg_sentence_count'], ref_stats['avg_sentence_count'])
    return {
        'length_similarity': length_similarity,
        'count_similarity': count_similarity
    }

def calculate_weighted_score(metrics: dict, weights: dict) -> float:
    """Calculate weighted score from metrics using provided weights."""
    return float(
        metrics['fre_similarity'] * weights['fre_similarity'] +
        metrics['sentence_length_similarity'] * weights['sentence_length_similarity'] +
        metrics['sentence_count_similarity'] * weights['sentence_count_similarity'] +
        metrics['rouge1'] * weights['rouge1'] +
        metrics['rougeS'] * weights['rougeS'] +
        metrics['punctuation_similarity'] * weights['punctuation_similarity']
    )

# Main function to compute all lexicality metrics
def lexicality(predictions: list[str], references: list[str]) -> dict:
    """
    Compute lexicality metrics comparing predictions with references.
    Returns averaged individual-level metrics and population-level similarity.
    
    Args:
        predictions: List of prediction texts
        references: List of reference texts
    """
    # Initialize lists to store individual metrics
    fre_similarities = []
    rouge1_scores = []
    rougeS_scores = []
    
    # Calculate individual-level metrics for each prediction-reference pair
    for pred, ref in zip(predictions, references):
        # Calculate basic metrics for prediction
        pred_fre = flesch_reading_ease(pred)
        rouge_1_f1, rouge_s_f1 = rouge_scores(pred, ref)
        
        # Calculate basic metrics for reference
        ref_fre = flesch_reading_ease(ref)
        
        # Calculate normalized individual-level metrics
        fre_similarities.append(relative_difference(pred_fre, ref_fre))
        rouge1_scores.append(rouge_1_f1)
        rougeS_scores.append(rouge_s_f1)
    
    # Calculate population-level metrics
    pred_sentence_stats = calculate_population_sentence_metrics(predictions)
    ref_sentence_stats = calculate_population_sentence_metrics(references)
    sentence_similarities = calculate_sentence_metrics_similarity(pred_sentence_stats, ref_sentence_stats)
    
    # Calculate population-level punctuation similarity
    pred_punct_dist = calculate_population_punct_distribution(predictions)
    ref_punct_dist = calculate_population_punct_distribution(references)
    punct_similarity = calculate_punct_distribution_similarity(pred_punct_dist, ref_punct_dist)
    
    # Calculate individual metrics
    metrics = {
        'fre_similarity': float(np.mean(fre_similarities)),
        'sentence_length_similarity': float(sentence_similarities['length_similarity']),
        'sentence_count_similarity': float(sentence_similarities['count_similarity']),
        'rouge1': float(np.mean(rouge1_scores)),
        'rougeS': float(np.mean(rougeS_scores)),
        'punctuation_similarity': float(punct_similarity)
    }
    
    # Calculate weighted score
    metrics['weighted_score'] = calculate_weighted_score(metrics, DEFAULT_WEIGHTS)
    
    return metrics
