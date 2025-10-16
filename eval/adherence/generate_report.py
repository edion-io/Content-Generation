"""Automatic Report Generation Script for Adherence Evaluator Evaluation.

This script generates a comprehensive README.md report analyzing the performance
of the adherence evaluator (LLM-as-judge) against human annotations.

The report includes:
1. Distribution of individual matching scores
2. Low-scoring cases at different thresholds (c_1=0.5, c_2=0.75)
3. LLM-based analysis of why samples scored low
4. LLM-based recommendations for improvement

Usage:
    python generate_report.py
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_report.log")
    ]
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive evaluation report for adherence evaluator."""

    def __init__(
        self,
        csv_path: str,
        json_path: str,
        output_path: str,
        openai_api_key: str | None = None
    ):
        """Initialize report generator.
        
        Args:
            csv_path: Path to CSV results file.
            json_path: Path to JSON results file.
            output_path: Path to output README.md file.
            openai_api_key: OpenAI API key (defaults to env var).

        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.output_path = output_path

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Load data
        self.df = pd.read_csv(csv_path)
        with open(json_path) as f:
            self.json_data = json.load(f)

        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")

    def calculate_thresholds(self) -> Dict[str, float]:
        """Calculate threshold scores based on the formula:
        score = (0.25 + 0.25 + 0.25 + 0.25 * c)
        
        Returns:
            Dictionary with threshold names and values.

        """
        # When only one field gets partial score (c * full_score)
        # and other three fields get full score
        # score = 0.25 * 1 + 0.25 * 1 + 0.25 * 1 + 0.25 * c
        #       = 0.75 + 0.25 * c

        thresholds = {
            "c_1": 0.75 + 0.25 * 0.5,   # c_1 = 1/2 → 0.875
            "c_2": 0.75 + 0.25 * 0.75,  # c_2 = 3/4 → 0.9375
        }

        logger.info(f"Calculated thresholds: {thresholds}")
        return thresholds

    def get_score_distribution(self) -> Dict[str, Any]:
        """Calculate score distribution statistics.
        
        Returns:
            Dictionary with distribution statistics.

        """
        scores = self.df["match_scores"].values

        distribution = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q1": float(np.percentile(scores, 25)),
            "q3": float(np.percentile(scores, 75)),
            "total_samples": len(scores),
        }

        # Count samples in different score ranges
        distribution["perfect_scores"] = int(np.sum(scores == 1.0))
        distribution["high_scores"] = int(np.sum((scores >= 0.9) & (scores < 1.0)))
        distribution["medium_scores"] = int(np.sum((scores >= 0.7) & (scores < 0.9)))
        distribution["low_scores"] = int(np.sum(scores < 0.7))

        logger.info(f"Score distribution: {distribution}")
        return distribution

    def get_low_scoring_cases(self, threshold: float) -> List[int]:
        """Get indices of cases with scores below threshold.
        
        Args:
            threshold: Score threshold.
            
        Returns:
            List of case indices (0-based).

        """
        low_cases = self.df[self.df["match_scores"] < threshold].index.tolist()
        logger.info(f"Found {len(low_cases)} cases below threshold {threshold}")
        return low_cases

    def parse_scores(self, score_str: str) -> Dict[str, Any]:
        """Parse score string to dictionary.
        
        Args:
            score_str: JSON string with scores.
            
        Returns:
            Dictionary with parsed scores.

        """
        try:
            return json.loads(score_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse score string: {score_str}")
            return {}

    def format_case_for_analysis(self, idx: int) -> str:
        """Format a single case for LLM analysis.
        
        Args:
            idx: Case index.
            
        Returns:
            Formatted case string.

        """
        row = self.df.iloc[idx]
        input_data = json.loads(row["inputs"])

        return f"""
Case {idx}:
- Match Score: {row['match_scores']:.4f}
- Query: {input_data['Query']}
- Answer: {input_data['Answer'][:200]}{'...' if len(input_data['Answer']) > 200 else ''}
- Reference Scores: {row['references']}
- Generated Scores: {row['generations']}
"""

    def analyze_low_scores_with_llm(
        self,
        low_case_indices: List[int],
        sample_size: int = 20
    ) -> str:
        """Use LLM to analyze why samples scored low.
        
        Args:
            low_case_indices: List of low-scoring case indices.
            sample_size: Number of cases to analyze (for API cost control).
            
        Returns:
            LLM-generated analysis text.

        """
        # Sample cases if too many
        if len(low_case_indices) > sample_size:
            sampled_indices = np.random.choice(
                low_case_indices,
                size=sample_size,
                replace=False
            ).tolist()
        else:
            sampled_indices = low_case_indices

        # Format cases for analysis
        cases_text = "\n".join([
            self.format_case_for_analysis(idx)
            for idx in sampled_indices
        ])

        prompt = f"""You are analyzing the performance of an LLM-based adherence evaluator (judge) that scores educational exercises.

The evaluator is supposed to match human reference annotations across 4 criteria:
- subject_score: 0/1 (binary)
- exercise_type_score: 0-4 (ordinal)
- grade_level_score: 0-4 (ordinal)
- modifier_score: 0-4 (ordinal)

The matching score is calculated as:
- For subject_score: 1.0 if equal, 0.0 otherwise
- For other scores: max(0, 1 - |generated - reference| / 4)
- Final score: weighted average of active fields (default weight 0.25 each)

Below are {len(sampled_indices)} low-scoring cases where the evaluator's judgments disagreed with human annotations.

{cases_text}

Please analyze these cases and identify:
1. Common patterns of mismatch (e.g., which criteria are most problematic)
2. Systematic biases (e.g., over/under-estimation tendencies)
3. Types of queries or answers that confuse the evaluator
4. Root causes of the mismatches

Provide a concise analysis (3-5 bullet points) focusing on actionable insights.
"""

        try:
            logger.info(f"Sending {len(sampled_indices)} cases to LLM for analysis...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in evaluating ML systems and identifying failure patterns."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )

            analysis = response.choices[0].message.content.strip()
            logger.info("LLM analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Error calling LLM for analysis: {e}")
            return "Error generating analysis. Please check logs."

    def generate_recommendations_with_llm(self, analysis: str) -> str:
        """Use LLM to generate recommendations based on analysis.
        
        Args:
            analysis: Analysis text from previous step.
            
        Returns:
            LLM-generated recommendations text.

        """
        prompt = f"""Based on the following analysis of an LLM-based adherence evaluator's performance:

{analysis}

Please provide specific, actionable recommendations to improve the evaluator's matching accuracy with human annotations.

Focus on:
1. Prompt engineering improvements
2. Schema and output format refinements
3. Few-shot examples or calibration strategies
4. Handling edge cases and ambiguous scenarios
5. Technical improvements (temperature, sampling, etc.)

Provide 5-8 concrete recommendations in bullet point format.
"""

        try:
            logger.info("Generating recommendations with LLM...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in prompt engineering and LLM system optimization."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            recommendations = response.choices[0].message.content.strip()
            logger.info("Recommendations generated")
            return recommendations

        except Exception as e:
            logger.error(f"Error calling LLM for recommendations: {e}")
            return "Error generating recommendations. Please check logs."

    def generate_markdown_table(
        self,
        case_indices: List[int],
        max_rows: int = 50
    ) -> str:
        """Generate markdown table for low-scoring cases.
        
        Args:
            case_indices: List of case indices.
            max_rows: Maximum number of rows to include.
            
        Returns:
            Markdown table string.

        """
        # Limit rows
        if len(case_indices) > max_rows:
            case_indices = case_indices[:max_rows]
            truncated = True
        else:
            truncated = False

        # Table header
        table = "| Index | Score | Query | Answer | Reference Scores | Generated Scores |\n"
        table += "|---:|---:|---|---|---|---|\n"

        # Table rows
        for idx in case_indices:
            row = self.df.iloc[idx]
            input_data = json.loads(row["inputs"])

            # Clean text for markdown (escape pipes, remove newlines)
            query = input_data["Query"].replace("|", "\\|").replace("\n", "<br>")[:150]
            answer = input_data["Answer"].replace("|", "\\|").replace("\n", "<br>")[:150]
            ref_scores = row["references"].replace("|", "\\|").replace("\n", " ")
            gen_scores = row["generations"].replace("|", "\\|").replace("\n", " ")

            table += f"| {idx} | {row['match_scores']:.4f} | {query}... | {answer}... | {ref_scores} | {gen_scores} |\n"

        if truncated:
            table += f"\n*Note: Table truncated to {max_rows} rows. Total low-scoring cases: {len(case_indices)}*\n"

        return table

    def generate_report(self) -> None:
        """Generate the complete README report."""
        logger.info("Starting report generation...")

        # Calculate statistics
        thresholds = self.calculate_thresholds()
        distribution = self.get_score_distribution()

        # Get low-scoring cases for each threshold
        low_cases_c1 = self.get_low_scoring_cases(thresholds["c_1"])
        low_cases_c2 = self.get_low_scoring_cases(thresholds["c_2"])

        # Analyze with LLM (use c_1 threshold cases)
        analysis = self.analyze_low_scores_with_llm(low_cases_c1, sample_size=20)
        recommendations = self.generate_recommendations_with_llm(analysis)

        # Generate markdown tables
        table_c1 = self.generate_markdown_table(low_cases_c1, max_rows=30)
        table_c2 = self.generate_markdown_table(low_cases_c2, max_rows=30)

        # Build the report
        report = self._build_report_content(
            distribution=distribution,
            thresholds=thresholds,
            low_cases_c1=low_cases_c1,
            low_cases_c2=low_cases_c2,
            table_c1=table_c1,
            table_c2=table_c2,
            analysis=analysis,
            recommendations=recommendations
        )

        # Write to file
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Report generated successfully: {self.output_path}")

    def _build_report_content(
        self,
        distribution: Dict[str, Any],
        thresholds: Dict[str, float],
        low_cases_c1: List[int],
        low_cases_c2: List[int],
        table_c1: str,
        table_c2: str,
        analysis: str,
        recommendations: str
    ) -> str:
        """Build the complete report content.
        
        Args:
            distribution: Score distribution statistics.
            thresholds: Calculated thresholds.
            low_cases_c1: Low-scoring case indices for c_1.
            low_cases_c2: Low-scoring case indices for c_2.
            table_c1: Markdown table for c_1 cases.
            table_c2: Markdown table for c_2 cases.
            analysis: LLM-generated analysis.
            recommendations: LLM-generated recommendations.
            
        Returns:
            Complete markdown report string.

        """
        overall_score = self.json_data["score"]
        total_samples = distribution["total_samples"]

        report = f"""### Adherence Evaluator – Evaluation Report

#### What is being evaluated
- **Goal**: Measure how well an adherence evaluator (LLM-as-judge) produces structured adherence scores that match human/reference annotations.
- **Criteria** (fields expected in structured output):
  - **subject_score**: 0/1 match for subject correctness
  - **exercise_type_score**: 0–4 adherence to requested exercise type/format
  - **grade_level_score**: 0–4 adherence to target grade level
  - **modifier_score**: 0–4 adherence to additional constraints/modifiers

#### Scoring method (theory)
- Each example has a reference JSON (ground truth scores) and a generated JSON (scores predicted by the evaluator).
- For each active field (present in reference):
  - subject_score is compared binary: 1.0 if equal else 0.0.
  - exercise_type_score, grade_level_score, modifier_score use a normalized distance on a 0–4 scale:
    - per-field match = max(0, 1 − |generated − reference| / 4)
  - final score = sum(per-field match * weight_i), where weight_i in default is 0.25 for each field.
- Fields that are missing in the reference are excluded from scoring; if present in reference but missing in generation → penalized as 0 for that field.
- Active field weights are normalized before aggregation. With default weights all equal, the final score is:
  - \\(\\text{{score}} = \\sum_{{f\\in \\text{{active}}}} w_f^{{(norm)}} \\cdot \\text{{match}}_f\\)

#### How it is implemented (from code)
- `AdherenceMatchEvaluator._evaluate_pair` parses reference/generated, computes per-field scores, selects active weights, normalizes, then aggregates to a final \\([0,1]\\) score.
- `Dataset(..., load_from_annotation=True)` builds inputs as a JSON with `{{"Query": ..., "Answer": ...}}` and references as a JSON containing the target scores.
- Generation uses `OpenAIGenerator` with a strict `SYSTEM_PROMPT` and a Pydantic `AdherenceSchema` to validate the structured JSON response.

#### How to run the evaluation
1) Prepare/merge annotation dataset (as noted in module header):
```bash
python /home/dingkang/projects/Content-Generation/eval/adherence/annotation_merge.py
```
2) Run the evaluator (uses `merged_annotation_dataset.csv` in this folder):
```bash
python /home/dingkang/projects/Content-Generation/eval/adherence/eval_adherence_evaluator.py
```
Outputs:
- JSON: `eval_adherence_evaluator_results.json` with `score`, `individual_scores`, `generations`, `inputs`, `references`
- CSV: `eval_adherence_evaluator_results.csv` (optional, if `save_scores=True`)

3) Generate this report:
```bash
python /home/dingkang/projects/Content-Generation/eval/adherence/generate_report.py
```

---

### Results Overview

#### Overall Performance
- **Overall Matching Score**: {overall_score:.4f}
- **Total Samples**: {total_samples}

| Metric | Value |
|---|---|
| Overall Score | {overall_score:.4f} |
| Total Samples | {total_samples} |
| Mean Score | {distribution['mean']:.4f} |
| Std Dev | {distribution['std']:.4f} |
| Median | {distribution['median']:.4f} |
| Min Score | {distribution['min']:.4f} |
| Max Score | {distribution['max']:.4f} |

---

### Distribution of Individual Scores

The distribution of matching scores across all {total_samples} samples:

| Score Range | Count | Percentage |
|---|---:|---:|
| Perfect (1.0) | {distribution['perfect_scores']} | {100 * distribution['perfect_scores'] / total_samples:.1f}% |
| High (0.9 - 0.99) | {distribution['high_scores']} | {100 * distribution['high_scores'] / total_samples:.1f}% |
| Medium (0.7 - 0.89) | {distribution['medium_scores']} | {100 * distribution['medium_scores'] / total_samples:.1f}% |
| Low (< 0.7) | {distribution['low_scores']} | {100 * distribution['low_scores'] / total_samples:.1f}% |

**Statistical Summary:**
- **Q1 (25th percentile)**: {distribution['q1']:.4f}
- **Q2 (50th percentile / Median)**: {distribution['median']:.4f}
- **Q3 (75th percentile)**: {distribution['q3']:.4f}

---

### Threshold Analysis

We define two thresholds based on the scoring formula where only one field gets a partial match:

**Threshold Calculation:**
- When 3 fields match perfectly (score = 1.0) and 1 field has partial match (score = c):
  - Final score = 0.25 × 1 + 0.25 × 1 + 0.25 × 1 + 0.25 × c = 0.75 + 0.25c

**Defined Thresholds:**
1. **c₁ = 0.5** (one field at 50% match): threshold = {thresholds['c_1']:.4f}
2. **c₂ = 0.75** (one field at 75% match): threshold = {thresholds['c_2']:.4f}

#### Low-Scoring Cases Summary

| Threshold | Value | Cases Below Threshold | Percentage |
|---|---:|---:|---:|
| c₁ (0.5) | {thresholds['c_1']:.4f} | {len(low_cases_c1)} | {100 * len(low_cases_c1) / total_samples:.1f}% |
| c₂ (0.75) | {thresholds['c_2']:.4f} | {len(low_cases_c2)} | {100 * len(low_cases_c2) / total_samples:.1f}% |

---

### Low-Scoring Cases (< {thresholds['c_1']:.4f}, c₁ threshold)

Cases where the evaluator's matching score is below {thresholds['c_1']:.4f} (indicating at least one field with ≤50% match or multiple fields with mismatches):

{table_c1}

---

### Low-Scoring Cases (< {thresholds['c_2']:.4f}, c₂ threshold)

Cases where the evaluator's matching score is below {thresholds['c_2']:.4f} (indicating at least one field with ≤75% match or multiple fields with mismatches):

{table_c2}

---

### Why These Samples Scored Low (Analysis)

**Context:** These are low matching scores between the evaluator's judgments and the reference annotations—not the original QA adherence itself. They reflect where the LLM-as-judge disagreed with human labels.

{analysis}

---

#### Recommendations (to Improve Matching Accuracy)

{recommendations}

---

*Report generated automatically by `generate_report.py`*
*Data source: `eval_adherence_evaluator_results.csv` and `eval_adherence_evaluator_results.json`*
"""

        return report


def main():
    """Main entry point for report generation."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "eval_adherence_evaluator_results.csv")
    json_path = os.path.join(script_dir, "eval_adherence_evaluator_results.json")
    output_path = os.path.join(script_dir, "README.md")

    # Check if input files exist
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please run eval_adherence_evaluator.py first to generate results.")
        return

    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        logger.error("Please run eval_adherence_evaluator.py first to generate results.")
        return

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        logger.error("Please set it to use LLM-based analysis and recommendations.")
        return

    # Generate report
    try:
        generator = ReportGenerator(
            csv_path=csv_path,
            json_path=json_path,
            output_path=output_path
        )
        generator.generate_report()
        logger.info("Report generation completed successfully!")

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

