import pandas as pd
import numpy as np
import json
import argparse
import time
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import the evaluation modules
from eval.adherence.adherence_eval import AdherenceEvaluator
from eval.lexicality.lexicality_eval import lexicality
from eval.semantic_similarity import SemanticSimilarity

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ExerciseGenEval:
    """
    Evaluation class for exercise generation models.
    
    Evaluates generated exercises using three metrics:
    1. Adherence - How well the exercise adheres to the user's instructions
    2. Lexicality - How well the exercise matches reference text style
    3. Semantic Similarity - How similar the generated content is to reference content
    """
    
    def __init__(self,
                 adherence_model: str = "gpt-4o-mini",
                 adherence_system_prompt: str = None,
                 adherence_temperature: float = 0.1,
                 semantic_models=None, 
                 semantic_modes=None,
                 semantic_weights=None,
                 semantic_power_transform=None):
        """
        Initialize the exercise generation evaluator.
        
        Args:
            adherence_model: Model to use for adherence evaluation
            adherence_system_prompt: System prompt to use for adherence evaluation
            adherence_temperature: Temperature to use for adherence evaluation
            semantic_models: List of models to use for semantic similarity
            semantic_modes: Pooling modes ("CLS" or "token_mean") for semantic models
            semantic_weights: Weights to combine similarity scores from multiple models.
            semantic_power_transform: Exponents to apply to individual similarity scores before weighting.
        """
        # Set up semantic similarity models
        if semantic_models is None:
            semantic_models = [
                "intfloat/multilingual-e5-large",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ]
        
        # Initialize semantic similarity evaluator
        self.semantic_sim = SemanticSimilarity(
            models=semantic_models,
            modes=semantic_modes,
            weights=semantic_weights,
            power_transform=semantic_power_transform
        )

        self.adherence_temperature = adherence_temperature

        self.adherence_evaluator = AdherenceEvaluator(
            generator_model=adherence_model,
            system_prompt=adherence_system_prompt
        )
    
    def evaluate_adherence(self, instructions: List[str], generations: List[str], masks: List[List[int]] = None) -> Dict:
        """
        Evaluate how well the generated exercise adheres to the instruction.
        
        Args:
            instructions: List of instructions given to generate the exercises
            generations: List of generated exercises
            masks: List of masks for each exercise, each mask is a list of 4 integers, 1 means the field is active, 0 means the field is inactive.
            
        Returns:
            Dict: Dictionary containing adherence scores for each dimension
        """
        print(masks)
        result = self.adherence_evaluator.adherence_score(instructions, generations, masks, temperature=self.adherence_temperature)
        print(result)
        return result
    
    def evaluate_lexicality(self, generations: List[str], references: List[str]) -> Dict:
        """
        Evaluate the lexical similarity between generated exercises and references.
        
        Args:
            generations: List of generated exercises
            references: List of reference exercises
            
        Returns:
            Dict: Dictionary containing lexicality metrics
        """
        return lexicality(generations, references)
    
    def evaluate_semantic_similarity(self, generations: List[str], references: List[str]) -> Dict:
        """
        Evaluate the semantic similarity between generated exercises and references.
        
        Args:
            generations: List of generated exercises
            references: List of reference exercises
            
        Returns:
            Dict: Dictionary containing semantic similarity score
        """
        similarities = []
        for gen, ref in tqdm(zip(generations, references), total=len(generations)):
            similarities.append(self.semantic_sim.get_similarity(gen, ref))
        
        return {
            "semantic_similarity": float(np.mean(similarities)),
            "individual_similarities": similarities
        }
    
    def load_data(self, file_path: str) -> Tuple[List[str], List[str], List[str], List[List[int]]|np.ndarray]:
        """
        Load data from a CSV or TSV file.
        
        Expected columns: instruction, generation, groundtruth
        Optional columns: subject, exercise_type, grade_level, modifier
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (instructions, generations, groundtruths, masks)
            masks: List[List[int]] or numpy array where each inner list contains masks for 
                   [subject, exercise_type, grade_level, modifier] columns
        """
        # Determine file format based on extension
        if file_path.endswith('.csv'):
            delimiter = ','
        elif file_path.endswith('.tsv'):
            delimiter = '\t'
        else:
            # Default to CSV
            print(f"Warning: Unknown file extension for {file_path}. Assuming CSV format.")
            delimiter = ','
        
        # Read the file
        try:
            data = pd.read_csv(file_path, delimiter=delimiter)
            
            # Check if required columns exist
            required_columns = ['instruction', 'generation', 'groundtruth']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Create masks based on empty param fields, where 1 if value is in mask_values, 0 otherwise
            mask_columns = ['subject', 'exercise_type', 'grade_level', 'modifier']
            mask_values = {'S', 'G', 'T', 'D', 'M'}

            if set(mask_columns) & set(data.columns):
                masks = data[mask_columns].isin(mask_values).astype(int).to_numpy()
            else:
                masks = np.zeros((len(data), len(mask_columns)))
            
            instructions = data['instruction'].tolist()
            generations = data['generation'].tolist()
            groundtruths = data['groundtruth'].tolist()
            
            return instructions, generations, groundtruths, masks
            
        except Exception as e:
            print(f"Error loading data file: {e}")
            raise
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary containing evaluation results
            output_path: Path to save the results
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
    
    def run(self, file_path: str, output_path: str = None) -> Dict:
        """
        Run the complete evaluation on a dataset.
        
        Args:
            file_path: Path to the data file
            output_path: Optional path to save results JSON
            
        Returns:
            Dict: Combined evaluation metrics
        """
        # Load the data
        instructions, generations, groundtruths, masks = self.load_data(file_path)

        # Calculate adherence scores
        print("Running adherence metric")
        adherence_results = self.evaluate_adherence(instructions, generations, masks)
        
        # Extract individual subscores
        subject_scores = [result.get("subject_score", 0) for result in adherence_results]
        exercise_type_scores = [result.get("exercise_type_score", 0) for result in adherence_results]
        grade_level_scores = [result.get("grade_level_score", 0) for result in adherence_results]
        modifier_scores = [result.get("modifier_score", 0) for result in adherence_results]
        total_scores = [result.get("overall_score", 0) for result in adherence_results]
        
        # Calculate lexicality metrics
        print("Running lexicality metric")
        lex_metrics = self.evaluate_lexicality(generations, groundtruths)
        
        # Calculate semantic similarity
        print("Running semantic similarity metric")
        sem_metrics = self.evaluate_semantic_similarity(generations, groundtruths)
        
        # Combine results
        results = {
            "adherence": {
                "mean_score": float(np.mean(total_scores)),
                "subject": {
                    "scores": subject_scores,
                    "mean_score": float(np.mean(subject_scores))
                },
                "exercise_type": {
                    "scores": exercise_type_scores,
                    "mean_score": float(np.mean(exercise_type_scores))
                },
                "grade_level": {
                    "scores": grade_level_scores,
                    "mean_score": float(np.mean(grade_level_scores))
                },
                "modifier": {
                    "scores": modifier_scores,
                    "mean_score": float(np.mean(modifier_scores))
                },
                "individual_scores": total_scores
            },
            "lexicality": {
                # Include all lexicality sub-metrics
                "fre_similarity": float(lex_metrics["fre_similarity"]),
                "sentence_length_similarity": float(lex_metrics["sentence_length_similarity"]),
                "sentence_count_similarity": float(lex_metrics["sentence_count_similarity"]),
                "rouge1": float(lex_metrics["rouge1"]),
                "rougeS": float(lex_metrics["rougeS"]),
                "punctuation_similarity": float(lex_metrics["punctuation_similarity"]),
                "weighted_score": float(lex_metrics["weighted_score"])
            },
            "semantic_similarity": {
                "mean_score": float(sem_metrics["semantic_similarity"]),
                "individual_scores": sem_metrics["individual_similarities"]
            },
            "overall_score": (
                float(np.mean(total_scores)) * 0.4 +  # Weight adherence by 40%
                lex_metrics["weighted_score"] * 0.3 +        # Weight lexicality by 30%
                sem_metrics["semantic_similarity"] * 0.3      # Weight semantic similarity by 30%
            )
        }
        
        # Save results if output path is provided
        if output_path:
            self.save_results(results, output_path)
        
        return results
    
    def generate_comparison_from_results_dict(self, model_eval_results: Dict[str, Dict], output_path: str = None) -> Dict:
        """
        Generate a comparison report from a dictionary of pre-computed model evaluation results.

        Args:
            model_eval_results: Dictionary mapping model names to their evaluation results
                                (output of the run() method).
            output_path: Optional path to save comparison results JSON.

        Returns:
            Dict: Comparative metrics for all models.
        """
        comparison = {
            "models": {},
            "comparison": {
                "overall_scores": {},
                "adherence_scores": {},
                "adherence_details": {
                    "subject": {},
                    "exercise_type": {},
                    "grade_level": {},
                    "modifier": {}
                },
                "lexicality_scores": {},
                "semantic_similarity_scores": {},
                "lexicality_details": {
                    "fre_similarity": {},
                    "sentence_length_similarity": {},
                    "sentence_count_similarity": {},
                    "rouge1": {},
                    "rougeS": {},
                    "punctuation_similarity": {}
                }
            }
        }

        # Extract relevant metrics for each model
        for model_name, results in model_eval_results.items():
            comparison["models"][model_name] = results  # Store the full result
            
            if not results: # Handle cases where a model might have failed evaluation
                print(f"Warning: No results found for model {model_name}. Skipping in comparison.")
                comparison["comparison"]["overall_scores"][model_name] = 0.0
                comparison["comparison"]["adherence_scores"][model_name] = 0.0
                comparison["comparison"]["adherence_details"]["subject"][model_name] = 0.0
                comparison["comparison"]["adherence_details"]["exercise_type"][model_name] = 0.0
                comparison["comparison"]["adherence_details"]["grade_level"][model_name] = 0.0
                comparison["comparison"]["adherence_details"]["modifier"][model_name] = 0.0
                comparison["comparison"]["lexicality_scores"][model_name] = 0.0
                comparison["comparison"]["semantic_similarity_scores"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["fre_similarity"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["sentence_length_similarity"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["sentence_count_similarity"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["rouge1"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["rougeS"][model_name] = 0.0
                comparison["comparison"]["lexicality_details"]["punctuation_similarity"][model_name] = 0.0
                continue

            comparison["comparison"]["overall_scores"][model_name] = results.get("overall_score", 0.0)
            
            adherence_data = results.get("adherence", {})
            comparison["comparison"]["adherence_scores"][model_name] = adherence_data.get("mean_score", 0.0)
            comparison["comparison"]["adherence_details"]["subject"][model_name] = adherence_data.get("subject", {}).get("mean_score", 0.0)
            comparison["comparison"]["adherence_details"]["exercise_type"][model_name] = adherence_data.get("exercise_type", {}).get("mean_score", 0.0)
            comparison["comparison"]["adherence_details"]["grade_level"][model_name] = adherence_data.get("grade_level", {}).get("mean_score", 0.0)
            comparison["comparison"]["adherence_details"]["modifier"][model_name] = adherence_data.get("modifier", {}).get("mean_score", 0.0)
            
            lexicality_data = results.get("lexicality", {})
            comparison["comparison"]["lexicality_scores"][model_name] = lexicality_data.get("weighted_score", 0.0)
            comparison["comparison"]["lexicality_details"]["fre_similarity"][model_name] = lexicality_data.get("fre_similarity", 0.0)
            comparison["comparison"]["lexicality_details"]["sentence_length_similarity"][model_name] = lexicality_data.get("sentence_length_similarity", 0.0)
            comparison["comparison"]["lexicality_details"]["sentence_count_similarity"][model_name] = lexicality_data.get("sentence_count_similarity", 0.0)
            comparison["comparison"]["lexicality_details"]["rouge1"][model_name] = lexicality_data.get("rouge1", 0.0)
            comparison["comparison"]["lexicality_details"]["rougeS"][model_name] = lexicality_data.get("rougeS", 0.0)
            comparison["comparison"]["lexicality_details"]["punctuation_similarity"][model_name] = lexicality_data.get("punctuation_similarity", 0.0)

            semantic_data = results.get("semantic_similarity", {})
            comparison["comparison"]["semantic_similarity_scores"][model_name] = semantic_data.get("mean_score", 0.0)

        # Calculate best model for each metric, handling empty scores
        def get_best_model(scores_dict: Dict[str, float]) -> str:
            if not scores_dict: return "N/A"
            return max(scores_dict.items(), key=lambda x: x[1])[0]

        best_overall = get_best_model(comparison["comparison"]["overall_scores"])
        best_adherence = get_best_model(comparison["comparison"]["adherence_scores"])
        best_subject = get_best_model(comparison["comparison"]["adherence_details"]["subject"])
        best_exercise_type = get_best_model(comparison["comparison"]["adherence_details"]["exercise_type"])
        best_grade_level = get_best_model(comparison["comparison"]["adherence_details"]["grade_level"])
        best_modifier = get_best_model(comparison["comparison"]["adherence_details"]["modifier"])
        best_lexicality = get_best_model(comparison["comparison"]["lexicality_scores"])
        best_semantic = get_best_model(comparison["comparison"]["semantic_similarity_scores"])
        
        comparison["best_models"] = {
            "overall": best_overall,
            "adherence": {
                "overall": best_adherence,
                "subject": best_subject,
                "exercise_type": best_exercise_type,
                "grade_level": best_grade_level,
                "modifier": best_modifier
            },
            "lexicality": best_lexicality,
            "semantic_similarity": best_semantic
        }

        # Save comparison if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"Comparison results saved to {output_path}")
        
        return comparison
    
    def visualize_results(self, results: Dict, output_path: str = None) -> None:
        """
        Create visualizations of evaluation results.
        
        Args:
            results: Dictionary containing evaluation results
            output_path: Optional path to save the visualization
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available. Install matplotlib and seaborn to enable this feature.")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Exercise Generation Evaluation Results', fontsize=16)
        
        # Plot 1: Overall score and component scores
        metrics = ['Overall', 'Adherence', 'Lexicality', 'Semantic']
        values = [
            results['overall_score'],
            results['adherence']['mean_score'],
            results['lexicality']['weighted_score'],
            results['semantic_similarity']['mean_score']
        ]
        
        axs[0, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].set_title('Overall Evaluation Scores')
        axs[0, 0].set_ylabel('Score (0-1)')
        
        # Plot 2: Adherence subscores
        adherence_metrics = ['Subject', 'Exercise Type', 'Grade Level', 'Modifier']
        adherence_values = [
            results['adherence']['subject']['mean_score'],
            results['adherence']['exercise_type']['mean_score'],
            results['adherence']['grade_level']['mean_score'],
            results['adherence']['modifier']['mean_score']
        ]
        
        axs[0, 1].bar(adherence_metrics, adherence_values, color='green')
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].set_title('Adherence Component Scores')
        axs[0, 1].set_ylabel('Score (0-1)')
        plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Lexicality component metrics
        lex_metrics = [
            results['lexicality']['fre_similarity'],
            results['lexicality']['sentence_length_similarity'],
            results['lexicality']['sentence_count_similarity'],
            results['lexicality']['rouge1'],
            results['lexicality']['rougeS'],
            results['lexicality']['punctuation_similarity']
        ]
        lex_labels = [
            'FRE Sim', 
            'Sent Length', 
            'Sent Count', 
            'ROUGE-1', 
            'ROUGE-S', 
            'Punct Sim'
        ]
        
        axs[1, 0].bar(lex_labels, lex_metrics, color='orange')
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_title('Lexicality Component Scores')
        axs[1, 0].set_ylabel('Score (0-1)')
        plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Weights visualization
        weights = {
            'Adherence': 0.4,
            'Lexicality': 0.3,
            'Semantic': 0.3
        }
        
        axs[1, 1].pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', 
                    startangle=90, colors=['green', 'orange', 'red'])
        axs[1, 1].set_title('Evaluation Weight Distribution')
        
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def visualize_comparison(self, comparison: Dict, output_path: str = None) -> None:
        """
        Create visualizations comparing multiple models.
        
        Args:
            comparison: Dictionary containing comparison results
            output_path: Optional path to save the visualization
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available. Install matplotlib and seaborn to enable this feature.")
            return
        
        # Create three plots: main metrics, adherence details, and lexicality details
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 20))
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        # Extract data for main metrics plotting
        model_names = list(comparison['comparison']['overall_scores'].keys())
        metrics = ['Overall', 'Adherence', 'Lexicality', 'Semantic']
        
        data_main = {
            'Model': [],
            'Metric': [],
            'Score': []
        }
        
        for model in model_names:
            # Add overall scores
            data_main['Model'].append(model)
            data_main['Metric'].append('Overall')
            data_main['Score'].append(comparison['comparison']['overall_scores'][model])
            
            # Add adherence scores (normalized to 0-1)
            data_main['Model'].append(model)
            data_main['Metric'].append('Adherence')
            data_main['Score'].append(comparison['comparison']['adherence_scores'][model] / 4.0)
            
            # Add lexicality scores
            data_main['Model'].append(model)
            data_main['Metric'].append('Lexicality')
            data_main['Score'].append(comparison['comparison']['lexicality_scores'][model])
            
            # Add semantic similarity scores
            data_main['Model'].append(model)
            data_main['Metric'].append('Semantic')
            data_main['Score'].append(comparison['comparison']['semantic_similarity_scores'][model])
        
        # Create data for adherence details
        data_adherence = {
            'Model': [],
            'Metric': [],
            'Score': []
        }
        
        adh_metrics = [
            'subject',
            'exercise_type',
            'grade_level',
            'modifier'
        ]
        
        adh_labels = [
            'Subject', 
            'Exercise Type', 
            'Grade Level', 
            'modifier'
        ]
        
        for model in model_names:
            for metric, label in zip(adh_metrics, adh_labels):
                data_adherence['Model'].append(model)
                data_adherence['Metric'].append(label)
                data_adherence['Score'].append(comparison['comparison']['adherence_details'][metric][model])
        
        # Create data for lexicality details
        data_lex = {
            'Model': [],
            'Metric': [],
            'Score': []
        }
        
        lex_metrics = [
            'fre_similarity',
            'sentence_length_similarity',
            'sentence_count_similarity',
            'rouge1',
            'rougeS',
            'punctuation_similarity'
        ]
        
        lex_labels = [
            'FRE Sim', 
            'Sent Length', 
            'Sent Count', 
            'ROUGE-1', 
            'ROUGE-S', 
            'Punct Sim'
        ]
        
        for model in model_names:
            for metric, label in zip(lex_metrics, lex_labels):
                data_lex['Model'].append(model)
                data_lex['Metric'].append(label)
                data_lex['Score'].append(comparison['comparison']['lexicality_details'][metric][model])
        
        # Create DataFrames
        df_main = pd.DataFrame(data_main)
        df_adherence = pd.DataFrame(data_adherence)
        df_lex = pd.DataFrame(data_lex)
        
        # Create grouped bar chart for main metrics
        sns.barplot(x='Metric', y='Score', hue='Model', data=df_main, ax=ax1)
        
        # Customize plot
        ax1.set_title('Main Evaluation Metrics', fontsize=14)
        ax1.set_xlabel('Evaluation Metric', fontsize=12)
        ax1.set_ylabel('Score (0-1)', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.legend(title='Model')
        
        # Add the "best model" labels above each group
        for i, metric in enumerate(metrics):
            if metric == 'Adherence':
                best_model = comparison['best_models']['adherence']['overall']
            elif metric == 'Overall':
                best_model = comparison['best_models']['overall']
            elif metric == 'Lexicality':
                best_model = comparison['best_models']['lexicality']
            else:  # Semantic
                best_model = comparison['best_models']['semantic_similarity']
                
            ax1.text(i, 0.95, f"Best: {best_model}", ha='center', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Create grouped bar chart for adherence details
        sns.barplot(x='Metric', y='Score', hue='Model', data=df_adherence, ax=ax2)
        
        # Customize plot
        ax2.set_title('Adherence Detail Metrics', fontsize=14)
        ax2.set_xlabel('Adherence Component', fontsize=12)
        ax2.set_ylabel('Score (0-1)', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend(title='Model')
        
        # Add the "best model" labels above each adherence detail
        for i, metric in enumerate(adh_labels):
            metric_key = metric.lower().replace(' ', '_')
            best_model = comparison['best_models']['adherence'].get(metric_key, '')
            ax2.text(i, 0.95, f"Best: {best_model}", ha='center', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Create grouped bar chart for lexicality details
        sns.barplot(x='Metric', y='Score', hue='Model', data=df_lex, ax=ax3)
        
        # Customize plot
        ax3.set_title('Lexicality Detail Metrics', fontsize=14)
        ax3.set_xlabel('Lexicality Component', fontsize=12)
        ax3.set_ylabel('Score (0-1)', fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.legend(title='Model')
        
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Simple test case
    import tempfile
    # import os # os is already imported at the top

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate exercise generation models')
    parser.add_argument('--input', type=str, help='Input file path (CSV or TSV) for single model evaluation.')
    parser.add_argument('--inputs_file', type=str, help='Path to a TXT file listing models and their data files (e.g., "model_name,path/to/data.csv" per line) for multi-model evaluation.')
    parser.add_argument('--output', type=str, help='Output file path for results (JSON). For --inputs_file, this saves a list of results. For --input, it saves a single result.')
    parser.add_argument('--test', action='store_true', help='Run with built-in test data for a single evaluation.')
    parser.add_argument('--compare', action='store_true', help='If --inputs_file is used, generate and save a comparison report of the models.')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations of results. Works with --input, --inputs_file (if --compare is also active), or --test.')
    parser.add_argument('--viz-output', type=str, help='Output path for visualization file (e.g., plot.png).')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ExerciseGenEval(
        adherence_model="gpt-4o-mini",
        adherence_temperature=0.1,
    )
    results = None # Initialize results to handle cases where it might not be set
    comparison_summary = None # Initialize for similar reasons

    if args.inputs_file:
        all_run_details = []
        model_eval_results_for_comparison = {}

        try:
            with open(args.inputs_file, 'r', encoding='utf-8') as f_inputs:
                for line in f_inputs:
                    line = line.strip()
                    if not line or ',' not in line:
                        if line: print(f"Skipping malformed line: {line}")
                        continue

                    parts = line.split(',', 1)
                    model_name = parts[0].strip()
                    csv_path = parts[1].strip()

                    print(f"Processing model: {model_name}, file: {csv_path}")
                    start_time = time.time()
                    try:
                        # Pass None for output_path to run, so it doesn't save individual JSONs here
                        eval_result = evaluator.run(csv_path, output_path=None)
                    except Exception as e:
                        print(f"Error evaluating model {model_name} from {csv_path}: {e}")
                        eval_result = None # Ensure eval_result is None on error

                    end_time = time.time()
                    run_time = end_time - start_time

                    all_run_details.append({
                        "model": model_name,
                        "input_file": csv_path,
                        "run_time": run_time,
                        "eval_result": eval_result
                    })
                    if eval_result: # Only add to comparison if evaluation was successful
                        model_eval_results_for_comparison[model_name] = eval_result
        except FileNotFoundError:
            print(f"Error: Inputs file not found at {args.inputs_file}")
            exit(1)
        except Exception as e:
            print(f"Error processing inputs file {args.inputs_file}: {e}")
            exit(1)

        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f_out:
                    json.dump(all_run_details, f_out, indent=2, ensure_ascii=False)
                print(f"Aggregated results for all models saved to {args.output}")
            except Exception as e:
                print(f"Error saving aggregated results to {args.output}: {e}")

        if args.compare and model_eval_results_for_comparison:
            print("\nGenerating model comparison report...")
            comparison_output_json_path = None
            if args.output:
                base, ext = os.path.splitext(args.output)
                comparison_output_json_path = f"{base}.comparison{ext}"
            else:
                comparison_output_json_path = "model_comparison_results.json"

            try:
                comparison_summary = evaluator.generate_comparison_from_results_dict(
                    model_eval_results_for_comparison,
                    output_path=comparison_output_json_path
                )
                print("\nModel Comparison Summary:")
                print("========================")
                if comparison_summary and comparison_summary.get('best_models'):
                    best_models = comparison_summary['best_models']
                    print(f"Best overall model: {best_models.get('overall', 'N/A')}")
                    if isinstance(best_models.get('adherence'), dict):
                        print(f"Best adherence: {best_models['adherence'].get('overall', 'N/A')}")
                    else:
                        print(f"Best adherence: N/A") # Default if structure is unexpected
                    print(f"Best lexicality: {best_models.get('lexicality', 'N/A')}")
                    print(f"Best semantic similarity: {best_models.get('semantic_similarity', 'N/A')}")
                else:
                    print("Could not determine best models from comparison summary.")

                if args.visualize and comparison_summary:
                    evaluator.visualize_comparison(comparison_summary, args.viz_output)
            except Exception as e:
                print(f"Error generating comparison report: {e}")
        elif args.compare and not model_eval_results_for_comparison:
            print("Skipping comparison: No successful model evaluations to compare.")

    elif args.input:
        print(f"Running evaluation for input file: {args.input}")
        try:
            start_time = time.time()
            results = evaluator.run(args.input, args.output)
            end_time = time.time()
            run_time = end_time - start_time
            print("\nEvaluation Results:")
            if results:
                print(f"Overall Score: {results.get('overall_score', 0.0):.4f}")
                print(f"Adherence Score: {results.get('adherence', {}).get('mean_score', 0.0):.4f}")
                print(f"Lexicality Score: {results.get('lexicality', {}).get('weighted_score', 0.0):.4f}")
                print(f"Semantic Similarity Score: {results.get('semantic_similarity', {}).get('mean_score', 0.0):.4f}")

                if args.visualize:
                    evaluator.visualize_results(results, args.viz_output)
            else:
                print("Evaluation did not produce results.")
        except Exception as e:
            print(f"Error during evaluation of {args.input}: {e}")

        if args.output:
            evl_result = [{
                "model": "default",
                "input_file": args.input,
                "run_time": run_time,
                "eval_result": results
            }]
            try:
                with open(args.output, 'w', encoding='utf-8') as f_out:
                    json.dump(evl_result, f_out, indent=2, ensure_ascii=False)
                print(f"Evaluation results saved to {args.output}")
            except Exception as e:
                print(f"Error saving evaluation results to {args.output}: {e}")

    elif args.test:
        print("Running with built-in test data...")
        test_data = """instruction,generation,groundtruth
"Create a math exercise for 5th grade on fractions.","Write the following fractions in order from smallest to largest: 3/4, 2/3, 5/6. Explain how you determined the order.","Order these fractions from smallest to largest: 2/3, 3/4, 5/6. Explain your method and reasoning."
"Generate a science question about the solar system for 3rd grade.","How many continents on earth?","List all planets in our solar system starting from the one closest to the sun."
"Create a language arts exercise about adjectives for 4th grade.","How many chemical elements?","Find all adjectives in this sentence: The large, brown dog ran after the tiny, orange cat."
"""

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write(test_data)
                temp_file_path = f.name

            results = evaluator.run(temp_file_path, args.output) # args.output can be None

            print("\nTest Evaluation Results:")
            if results:
                print(f"Overall Score: {results.get('overall_score', 0.0):.4f}")
                print(f"Adherence Score: {results.get('adherence', {}).get('mean_score', 0.0):.4f}")
                print(f"Lexicality Score: {results.get('lexicality', {}).get('weighted_score', 0.0):.4f}")
                print(f"Semantic Similarity Score: {results.get('semantic_similarity', {}).get('mean_score', 0.0):.4f}")

                if args.visualize:
                    evaluator.visualize_results(results, args.viz_output)
            else:
                print("Test evaluation did not produce results.")
        except Exception as e:
            print(f"Error during test run: {e}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
