"""
This script evaluates the reliability of the adherence evaluator.

Quadratic Weighted Kappa (QWK) - is chosen as the best single metric because:
It accounts for the ordinal nature of 0-4 scale
It penalizes disagreements proportionally to their distance (disagreeing by 3 points is worse than disagreeing by 1)
It corrects for chance agreement
Values: -1 to 1, where 1 = perfect agreement, 0 = chance agreement

Mean Absolute Error (MAE) - Simple and interpretable:
Directly measures average prediction error
Easy to understand: "On average, the evaluator is off by X points"

MAE is often more interpretable for discrete scales, 
and QWK is more appropriate for ordinal inter-rater reliability assessment.
"""

from typing import Dict, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, 
    cohen_kappa_score, 
    f1_score, 
    matthews_corrcoef, 
    precision_score, 
    recall_score,
    classification_report, 
    confusion_matrix,
    mean_absolute_error,
    )

from scipy.stats import pearsonr, spearmanr
import numpy as np
from numpy.typing import ArrayLike


def evaluate_ordinal_adherence_reliability(
    human_scores: ArrayLike, 
    evaluator_scores: ArrayLike
) -> Dict[str, float]:
    """
    Evaluate reliability of ordinal adherence evaluator
    
    Args:
        human_scores: Ground truth scores (0-4)
        evaluator_scores: Evaluator predicted scores (0-4)
        
    Returns:
        Dictionary containing reliability metrics:
        - MAE: Mean Absolute Error
        - Quadratic_Weighted_Kappa: Quadratic weighted Cohen's kappa
        - Pearson_Correlation: Pearson correlation coefficient
        - Spearman_Correlation: Spearman rank correlation coefficient
    """
    # Primary metrics
    mae = mean_absolute_error(human_scores, evaluator_scores)
    
    # Quadratic weighted kappa
    qwk = cohen_kappa_score(human_scores, evaluator_scores, weights='quadratic')
    
    # Correlation metrics
    pearson_r, _ = pearsonr(human_scores, evaluator_scores)
    spearman_r, _ = spearmanr(human_scores, evaluator_scores)
    
    return {
        'MAE': mae,
        'Quadratic_Weighted_Kappa': qwk,
        'Pearson_Correlation': pearson_r,
        'Spearman_Correlation': spearman_r
    }

def evaluate_binary_adherence_reliability(
    human_binary: ArrayLike, 
    evaluator_binary: ArrayLike
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate reliability of binary adherence evaluator
    
    Args:
        human_binary: Ground truth binary scores (0/1)
        evaluator_binary: Evaluator predicted binary scores (0/1)
        
    Returns:
        Dictionary containing reliability metrics:
        - Accuracy: Classification accuracy
        - Cohen_Kappa: Cohen's kappa coefficient
        - F1_Score: F1 score
        - MCC: Matthews Correlation Coefficient
        - Precision: Precision score
        - Recall: Recall score
        - Confusion_Matrix: Confusion matrix as numpy array
    """
    # Primary metrics
    accuracy = accuracy_score(human_binary, evaluator_binary)
    kappa = cohen_kappa_score(human_binary, evaluator_binary)
    
    # Complementary metrics
    f1 = f1_score(human_binary, evaluator_binary)
    mcc = matthews_corrcoef(human_binary, evaluator_binary)
    precision = precision_score(human_binary, evaluator_binary)
    recall = recall_score(human_binary, evaluator_binary)
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(human_binary, evaluator_binary)
    
    return {
        'Accuracy': accuracy,
        'Cohen_Kappa': kappa,
        'F1_Score': f1,
        'MCC': mcc,
        'Precision': precision,
        'Recall': recall,
        'Confusion_Matrix': cm
    }

def comprehensive_adherence_evaluation(
    human_ordinal: ArrayLike, 
    evaluator_ordinal: ArrayLike, 
    human_binary: ArrayLike, 
    evaluator_binary: ArrayLike
) -> Tuple[Dict[str, float], Dict[str, Union[float, np.ndarray]]]:
    """
    Comprehensive evaluation for both ordinal (0-4) and binary (0/1) adherence scores
    
    Args:
        human_ordinal: Ground truth ordinal scores (0-4)
        evaluator_ordinal: Evaluator predicted ordinal scores (0-4)
        human_binary: Ground truth binary scores (0/1)
        evaluator_binary: Evaluator predicted binary scores (0/1)
        
    Returns:
        Tuple containing:
        - ordinal_results: Dictionary with ordinal reliability metrics
        - binary_results: Dictionary with binary reliability metrics
    """
    print("=== ORDINAL ADHERENCE SCORES (0-4) ===")
    ordinal_results = evaluate_ordinal_adherence_reliability(human_ordinal, evaluator_ordinal)
    for metric, value in ordinal_results.items():
        print(f"{metric}: {value:.3f}")
    
    print("\n=== BINARY ADHERENCE SCORES (0/1) ===")
    binary_results = evaluate_binary_adherence_reliability(human_binary, evaluator_binary)
    for metric, value in binary_results.items():
        if metric != 'Confusion_Matrix':
            print(f"{metric}: {value:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(binary_results['Confusion_Matrix'])
    
    return ordinal_results, binary_results