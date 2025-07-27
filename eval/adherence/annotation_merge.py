"""
Annotation Data Merger and Perfect Adherence Sample Generator

This script merges existing annotation datasets and generates additional samples 
with perfect adherence scores for balanced evaluation.

Features:
- Merges multiple annotation CSV files
- Handles missing fields gracefully
- Generates perfect adherence samples
- Ensures consistent schema across all data
- Clean, elegant, and type-annotated implementation

Usage:
    python annotation_merge.py
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np


class AnnotationMerger:
    """Elegant annotation dataset merger and sample generator."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize with project structure."""
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.eval_dir = self.project_root / "eval" / "adherence"
        
        # Define the standard schema for annotation data
        self.schema = {
            'original_id': 'Int64',
            'original_instruction': 'string',
            'original_subject': 'string', 
            'original_exercise_type': 'string',
            'original_difficulty': 'string',
            'original_grade_level': 'string',
            'original_modifier': 'string',
            'original_generation': 'string',
            'new_instruction': 'string',
            'new_subject': 'string',
            'new_exercise_type': 'string', 
            'new_difficulty': 'string',
            'new_grade_level': 'string',
            'new_modifier': 'string',
            # Adherence scores should be nullable integers (Int64)
            'adherence_subject': 'Int64',
            'adherence_exercise_type': 'Int64',
            'adherence_difficulty': 'Int64', 
            'adherence_grade_level': 'Int64',
            'adherence_modifier': 'Int64',
            'annotation_timestamp': 'string'
        }

        # Columns that hold adherence scores for easy reference elsewhere
        self._score_cols: List[str] = [
            'adherence_subject',
            'adherence_exercise_type',
            'adherence_difficulty',
            'adherence_grade_level',
            'adherence_modifier',
        ]
    
    def load_and_standardize(self, filepath: Path) -> pd.DataFrame:
        """Load CSV and standardize to schema, handling missing columns gracefully."""
        try:
            df = pd.read_csv(filepath)
            print(f"📂 Loaded {len(df)} records from {filepath.name}")
            
            # Add missing columns with null values
            for col, dtype in self.schema.items():
                if col not in df.columns:
                    df[col] = np.nan
                    print(f"  ➕ Added missing column: {col}")
            
            # Reorder and cast to schema (first cast non-score columns)
            df = df[list(self.schema.keys())]

            # Convert adherence score columns to numeric Int64 safely
            for col in self._score_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').round()
                df[col] = df[col].astype('Int64')

            # Cast remaining columns per schema (non-score columns may already be correct)
            non_score_cols = [c for c in self.schema.keys() if c not in self._score_cols]
            df[non_score_cols] = df[non_score_cols].astype({k: v for k, v in self.schema.items() if k in non_score_cols})
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return pd.DataFrame(columns=list(self.schema.keys())).astype(self.schema)
    
    def load_source_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the original questions and instructions datasets."""
        questions_path = self.data_dir / "questions.csv"
        instructions_path = self.data_dir / "instructions.csv"
        
        if not questions_path.exists():
            print(f"⚠️  Source data not found: {questions_path}")
            print(f"📁 Available directories: {list(self.project_root.iterdir())}")
            return pd.DataFrame(), pd.DataFrame()
            
        if not instructions_path.exists():
            print(f"⚠️  Source data not found: {instructions_path}")
            print(f"📁 Available files in data: {list(self.data_dir.iterdir()) if self.data_dir.exists() else 'data dir not found'}")
            return pd.DataFrame(), pd.DataFrame()
            
        questions = pd.read_csv(questions_path)
        instructions = pd.read_csv(instructions_path)
        
        print(f"📊 Loaded source data: {len(questions)} questions, {len(instructions)} instructions")
        return questions, instructions
    
    def generate_perfect_samples(self, n_samples: int = 40) -> pd.DataFrame:
        """Generate samples with perfect adherence (no parameter changes)."""
        questions, instructions = self.load_source_data()
        
        # Handle case where source data is not available
        if questions.empty or instructions.empty:
            print(f"⚠️  Cannot generate perfect samples: source data not available")
            return pd.DataFrame(columns=list(self.schema.keys())).astype(self.schema)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Sample random indices
        sample_indices = np.random.choice(len(questions), size=min(n_samples, len(questions)), replace=False)
        
        perfect_samples = []
        timestamp = datetime.now().isoformat()
        
        for idx in sample_indices:
            q_row = questions.iloc[idx]
            i_row = instructions.iloc[idx]
            
            # Create perfect adherence sample
            sample = {
                'original_id': int(idx),
                'original_instruction': i_row['Input'],
                'original_subject': q_row['subject'],
                'original_exercise_type': q_row['exercise_type'],
                'original_difficulty': q_row['difficulty'],
                'original_grade_level': str(q_row['grade_level']),
                'original_modifier': q_row['modifier'],
                'original_generation': q_row['content'],
                # Perfect adherence: new = original
                'new_instruction': i_row['Input'],
                'new_subject': q_row['subject'],
                'new_exercise_type': q_row['exercise_type'],
                'new_difficulty': q_row['difficulty'],
                'new_grade_level': str(q_row['grade_level']),
                'new_modifier': q_row['modifier'],
                # Perfect adherence scores
                'adherence_subject': 1,
                'adherence_exercise_type': 4,
                'adherence_difficulty': 4,
                'adherence_grade_level': 4,
                'adherence_modifier': 4,
                'annotation_timestamp': timestamp
            }
            perfect_samples.append(sample)
        
        df = pd.DataFrame(perfect_samples)

        # Ensure score columns are Int64
        for col in self._score_cols:
            df[col] = df[col].astype('Int64')

        # Cast remaining columns per schema
        non_score_cols = [c for c in self.schema.keys() if c not in self._score_cols]
        df[non_score_cols] = df[non_score_cols].astype({k: v for k, v in self.schema.items() if k in non_score_cols})

        print(f"✨ Generated {len(df)} perfect adherence samples")
        return df
    
    def mask_placeholder_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask adherence scores with NaN when their corresponding parameter contains a placeholder code.

        The dataset uses single-letter placeholder codes to indicate that a particular
        parameter (Subject, Grade-level, Exercise Type, Modifier, Difficulty)
        was *intentionally* left unspecified during generation. In these cases
        the associated adherence score is meaningless and should therefore be
        set to missing (NaN/NA).

        The mapping between parameters and their placeholder codes is:
            Subject        -> "S"
            Grade-level    -> "G"
            Exercise Type  -> "T"
            Modifier       -> "M"
            Difficulty     -> "D"
        """
        placeholder_map: Dict[str, str] = {
            "subject": "S",
            "grade_level": "G",
            "exercise_type": "T",
            "modifier": "M",
            "difficulty": "D",
        }

        for param, code in placeholder_map.items():
            original_col = f"original_{param}"
            new_col = f"new_{param}"
            adherence_col = f"adherence_{param if param != 'subject' else 'subject'}" if param != 'exercise_type' else "adherence_exercise_type"

            # Build a mask where either the original or the new value equals the placeholder code
            mask = df[original_col].astype(str).eq(code) | df[new_col].astype(str).eq(code)

            # Apply the mask to set the adherence score to NA/NaN
            df.loc[mask, adherence_col] = np.nan

        return df
    
    def merge_datasets(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Merge all annotation datasets and generate additional samples."""
        # Load existing annotation files
        existing_files = [
            self.eval_dir / "annotated_dataset.csv",
            self.eval_dir / "extra_adherence_annotations.csv"
        ]
        
        dataframes = []
        
        # Load and standardize existing files
        for filepath in existing_files:
            if filepath.exists():
                df = self.load_and_standardize(filepath)
                if not df.empty:
                    dataframes.append(df)
        
        # Generate perfect adherence samples
        perfect_df = self.generate_perfect_samples(31)
        dataframes.append(perfect_df)
        
        # Merge all dataframes
        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Handle duplicate IDs by keeping the latest annotation
            merged_df = merged_df.sort_values('annotation_timestamp', na_position='first')
            merged_df = merged_df.drop_duplicates(subset=['original_id'], keep='last')

            # Mask adherence scores where the corresponding parameters are placeholders
            merged_df = self.mask_placeholder_scores(merged_df)

            # Final enforcement: ensure adherence scores are Int64
            for col in self._score_cols:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').round().astype('Int64')

            print(f"🔗 Merged dataset contains {len(merged_df)} total records")
            print(f"📈 Distribution by adherence_subject: {merged_df['adherence_subject'].value_counts(dropna=False).to_dict()}")
            
            # Save merged dataset
            output_path = output_path or (self.eval_dir / "merged_annotation_dataset.csv")
            merged_df.to_csv(output_path, index=False)
            print(f"💾 Saved merged dataset to: {output_path}")
            
            return merged_df
        else:
            print("⚠️  No valid data found to merge")
            return pd.DataFrame(columns=list(self.schema.keys())).astype(self.schema)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the merged dataset and return quality metrics."""
        metrics = {
            'total_records': len(df),
            'complete_records': len(df.dropna()),
            'missing_original_data': df[['original_instruction', 'original_generation']].isnull().any(axis=1).sum(),
            'missing_adherence_scores': df[self._score_cols].isnull().any(axis=1).sum(),
            'adherence_score_distribution': {
                'subject': df['adherence_subject'].value_counts(dropna=False).to_dict(),
                'exercise_type': df['adherence_exercise_type'].value_counts(dropna=False).to_dict(),
                'difficulty': df['adherence_difficulty'].value_counts(dropna=False).to_dict(),
                'grade_level': df['adherence_grade_level'].value_counts(dropna=False).to_dict(),
                'modifier': df['adherence_modifier'].value_counts(dropna=False).to_dict()
            }
        }
        
        print("\n📊 Dataset Validation Summary:")
        print(f"   Total Records: {metrics['total_records']}")
        print(f"   Complete Records: {metrics['complete_records']}")
        print(f"   Missing Original Data: {metrics['missing_original_data']}")
        print(f"   Missing Adherence Scores: {metrics['missing_adherence_scores']}")
        
        return metrics


def main():
    """Main execution function."""
    print("🚀 Starting Annotation Dataset Merger\n")
    
    merger = AnnotationMerger()
    
    try:
        # Merge datasets and generate samples
        merged_df = merger.merge_datasets()
        
        # Validate the result
        metrics = merger.validate_data(merged_df)
        
        print("\n✅ Annotation merge completed successfully!")
        print(f"📁 Output: {merger.eval_dir / 'merged_annotation_dataset.csv'}")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise


if __name__ == "__main__":
    main() 