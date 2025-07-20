"""
Evaluation System for Adherence Evaluator Performance.

This module evaluates how well an adherence evaluator can generate accurate adherence scores
that match reference annotations. It tests the evaluator's ability to assess educational
exercises across multiple criteria including subject, exercise type, grade level, and modifiers.

Main Components:
- OpenAIGenerator: Interface for generating adherence scores using OpenAI API or compatible services
- AdherenceMatchEvaluator: Compares generated adherence scores against reference scores
- Dataset: Handles loading and batching of evaluation data from annotated CSV files
- Pydantic schemas for structured adherence scoring validation

The evaluation system measures:
- Binary scoring accuracy for subject adherence (0/1)
- Ordinal scoring accuracy for exercise type, grade level, and modifier adherence (0-4)
- Weighted overall performance across multiple criteria
- Batch processing for efficient evaluation
- Structured JSON output validation

Usage:
    python eval_adherence_evaluator.py

This will run the evaluation pipeline to test how well the adherence evaluator
performs against ground truth annotations in the dataset.
"""
import asyncio
import re
from typing import Iterator, List, Literal, Optional, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import json

import numpy as np
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import pandas as pd
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('eval_adherence.log')  # File output
    ]
)

logger = logging.getLogger(__name__)


class OpenAIGenerator:
    """Generator using OpenAI API or compatible APIs."""

    def __init__(self, model: str = "gpt-4o-mini", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7):
        """
        Initialize OpenAI generator.
        
        Args:
            model: Model name to use.
            api_key: OpenAI API key (defaults to env var OPENAI_API_KEY).
            base_url: Base URL for OpenAI compatible APIs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize clients
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        elif os.getenv('OPENAI_API_KEY'):
            client_kwargs['api_key'] = os.getenv('OPENAI_API_KEY')

        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def run(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        try:
            if kwargs.get('response_format', None) is not None:
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": kwargs.get("system_prompt", SYSTEM_PROMPT),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    response_format=kwargs.get("response_format", None),
                )
                return response.choices[0].message.parsed.model_dump_json()
            
            else:   
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature)                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise

    async def run_async(self, prompt: str, **kwargs) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: Input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        try:
            if kwargs.get('response_format', None) is not None:
                response = await self.async_client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": kwargs.get("system_prompt", SYSTEM_PROMPT),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    response_format=kwargs.get("response_format", None),
                )
                return response.choices[0].message.parsed.model_dump_json()
            
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    n=1
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI async: {e}")
            raise

    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts using async batch processing.
        
        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated texts.
        """
        async def _batch_generate():
            tasks = []
            # for prompt in tqdm(prompts, desc="Sending batch requests"):
            for prompt in prompts:
                task = self.run_async(prompt, **kwargs)
                tasks.append(task)
            return await asyncio.gather(*tasks)

        return asyncio.run(_batch_generate())


class _BinaryAdherenceScore(BaseModel):
    score: Literal[0, 1] = Field(description="The adherence score for binary adherence")
    explanation: str = Field(description="The explanation for the adherence score")

class _OrdinalAdherenceScore(BaseModel):
    score: Literal[0, 1, 2, 3, 4] = Field(description="The adherence score for ordinal adherence")
    explanation: str = Field(description="The explanation for the adherence score")

class _AdherenceWithExplanationSchema(BaseModel):
    """Structured schema for adherence LLM output.
    This model is used to enforce JSON mode responses when interacting
    with the generator. All fields are optional because the query may
    legitimately omit certain criteria (the correct value in that case
    is ``null`` / ``None``).
    """
    subject: _BinaryAdherenceScore = Field(description="The adherence score for the subject")
    exercise_type: _OrdinalAdherenceScore = Field(description="The adherence score for the exercise type")
    grade_level: _OrdinalAdherenceScore = Field(description="The adherence score for the grade level")
    modifier: _OrdinalAdherenceScore = Field(description="The adherence score for the modifier")

class _AdherenceSchema(BaseModel):
    """Structured schema for adherence LLM output.
    This model is used to enforce JSON mode responses when interacting
    with the generator. Fields that are not provided will default to None.
    """

    subject_score: Literal[0, 1] = Field(..., description="The adherence score for the subject, binary")
    exercise_type_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the exercise type, ordinal")
    grade_level_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the grade level, ordinal")
    modifier_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the modifier, ordinal")


class AdherenceMatchEvaluator:
    """
    Evaluator that measures how well a system can provide adherence scores
    that match reference adherence scores.
    
    This evaluator is designed for evaluating prompt tuning systems that
    generate adherence scores across multiple criteria.
    """
    
    def __init__(self, 
                 subject_weight: float = 0.25,
                 exercise_type_weight: float = 0.25,
                 grade_level_weight: float = 0.25,
                 modifier_weight: float = 0.25,):
        """
        Initialize adherence match evaluator.
        
        Args:
            subject_weight: Weight for subject matching score.
            exercise_type_weight: Weight for exercise type adherence score.
            grade_level_weight: Weight for grade level adherence score.
            modifier_weight: Weight for modifier adherence score.
        """
        self.subject_weight = subject_weight
        self.exercise_type_weight = exercise_type_weight
        self.grade_level_weight = grade_level_weight
        self.modifier_weight = modifier_weight
        
        # Ensure weights sum to 1.0
        total_weight = (subject_weight + exercise_type_weight + 
                       grade_level_weight + modifier_weight)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight}, not 1.0. "
                          "Consider normalizing weights.")
    
    def run(self, inputs: List[str], references: List[str], 
            generations: List[str]) -> float:
        """
        Evaluate adherence matching of generated scores to reference scores.
        
        Args:
            inputs: List of input prompts (not used directly).
            references: List of reference adherence scores.
            generations: List of generated adherence scores.
            
        Returns:
            Average adherence matching score (0.0 to 1.0, higher is better).
        """
        if len(references) != len(generations):
            raise ValueError(f"Number of references ({len(references)}) "
                           f"must match number of generations ({len(generations)})")
        
        if not references:
            return 0.0
        
        scores = []
        for ref, gen in zip(references, generations):
            score = self._evaluate_pair(ref, gen)
            scores.append(score)
        
        return float(np.mean(scores))
    
    def run_detailed(self, inputs: List[str], references: List[str], 
                    generated: List[str]) -> Tuple[float, List[float]]:
        """
        Evaluate adherence matching with detailed individual scores.
        
        Args:
            inputs: List of input prompts (not used directly).
            references: List of reference adherence scores.
            generated: List of generated adherence scores.
            
        Returns:
            Tuple of (overall_score, individual_scores).
        """
        if len(references) != len(generated):
            raise ValueError(f"Number of references ({len(references)}) "
                           f"must match number of generated ({len(generated)})")
        
        if not references:
            return 0.0, []
        
        individual_scores = []
        for ref, gen in zip(references, generated):
            score = self._evaluate_pair(ref, gen)
            if score is None:
                continue
            individual_scores.append(score)
        
        overall_score = float(np.mean(individual_scores))
        return overall_score, individual_scores
    
    def _evaluate_pair(self, reference: str, generated: str) -> float:
        """
        Evaluate a single reference-generated pair with masking support.
        
        Args:
            reference: Reference adherence scores.
            generated: Generated adherence scores.
            
        Returns:
            Adherence matching score for this pair.
        """
        try:
            # Parse the scores
            reference_scores = self._parse_response(reference)
            generated_scores = self._parse_response(generated)
            
            if reference_scores is None or generated_scores is None:
                return 0.0
            
            # Calculate component scores with None handling
            component_scores = {}
            active_weights = {}
            
            # Define field mappings
            field_mappings = {
                "subject_score": ("subject_weight", self.subject_weight),
                "exercise_type_score": ("exercise_type_weight", self.exercise_type_weight),
                "grade_level_score": ("grade_level_weight", self.grade_level_weight),
                "modifier_score": ("modifier_weight", self.modifier_weight)
            }
            
            for field, (weight_name, weight_value) in field_mappings.items():
                ref_value = reference_scores.get(field)
                gen_value = generated_scores.get(field)
                
                if ref_value is None:
                    # If reference is None, exclude from scoring
                    continue
                elif gen_value is None:
                    # If generated is None but reference is not, penalize
                    component_scores[field] = 0.0
                    active_weights[field] = weight_value
                else:
                    # Both have values, calculate score
                    if field == "subject_score":
                        # Binary comparison for subject score
                        component_scores[field] = 1.0 if gen_value == ref_value else 0.0
                    else:
                        # Continuous comparison for other scores (0-4 scale)
                        component_scores[field] = max(0.0, 1.0 - abs(gen_value - ref_value) / 4.0)
                    
                    active_weights[field] = weight_value
            
            # Check if any active weights exist
            if not active_weights:
                # All fields were None in reference, perfect match
                return 1.0
            
            # Normalize active weights to sum to 1.0
            total_active_weight = sum(active_weights.values())
            if total_active_weight == 0:
                return 1.0
            
            normalized_weights = {
                field: weight / total_active_weight 
                for field, weight in active_weights.items()
            }
            
            # Calculate weighted score
            total_score = sum(
                normalized_weights[field] * component_scores[field]
                for field in component_scores
            )
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error evaluating adherence match pair: {e}")
            return 0.0
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse response to extract adherence scores with None masking support.
        
        Args:
            response: Response string containing adherence scores.
            
        Returns:
            Dictionary with parsed scores (values can be None) or None if parsing fails.
        """
        try:
            # Extract JSON-like content from the response
            json_matches = re.findall(r"{[\s\S]+}", response)
            if not json_matches:
                return None
            
            response_json = json_matches[0]
            
            # Convert Python None to JSON null for proper parsing
            response_json = response_json.replace("None", "null").replace("NaN", "null")
            
            # Use json.loads for safety instead of eval
            parsed = json.loads(response_json)

            # If the response is a structured response, convert it to a dictionary
            if "subject" in parsed:
                parsed = {
                    "subject_score": parsed["subject"]["score"],
                    "exercise_type_score": parsed["exercise_type"]["score"],
                    "grade_level_score": parsed["grade_level"]["score"],
                    "modifier_score": parsed["modifier"]["score"]
                }
            
            # Validate required keys exist (values can be None)
            required_keys = [
                "subject_score", "exercise_type_score", "grade_level_score", "modifier_score"
            ]
            if not all(key in parsed for key in required_keys):
                logger.warning(f"Missing required keys in parsed response: {parsed}, should be {required_keys}")
                return None
            
            parsed = {k: int(v) for k, v in parsed.items() if v is not None}
            
            # Validate that non-None values are within expected ranges
            for key, value in parsed.items():
                if value is not None:
                    if key == "subject_score":
                        # subject_score should be a string or integer (0/1)
                        if isinstance(value, int):
                            if value not in [0, 1]:
                                logger.warning(f"Invalid subject_score value: {value}")
                                parsed[key] = None
                        elif not isinstance(value, str):
                            logger.warning(f"Invalid subject_score type: {type(value)}")
                            parsed[key] = None
                    elif key in ["exercise_type_score", "grade_level_score", "modifier_score"]:
                        # These should be numeric values between 0-4
                        if not isinstance(value, (int, float)) or not (0 <= value <= 4):
                            logger.warning(f"Invalid {key} value: {value}")
                            parsed[key] = None
            
            return parsed
            
        except (IndexError, json.JSONDecodeError, ValueError, SyntaxError) as e:
            import traceback
            traceback.print_exc()
            logger.info(f"response: {response}")
            logger.error(f"Error parsing response: {e}")
            return None

class Dataset:

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = False,
        input_column: str = "Inputs",
        reference_column: str = "Outputs",
        load_from_annotation: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_column = input_column
        self.reference_column = reference_column
        self._data: Optional[pd.DataFrame] = None
        if load_from_annotation:
            self._load_data_from_annotation()
        else:
            self._load_data()

    def _load_data(self) -> None:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(self.data_path)

            # Validate required columns
            if self.input_column not in df.columns:
                raise ValueError(f"Input column '{self.input_column}' "
                               f"not found in CSV")
            if self.reference_column not in df.columns:
                raise ValueError(f"Reference column '{self.reference_column}' "
                               f"not found in CSV")

            # Create standardized dataframe
            self._data = pd.DataFrame({
                'input': df[self.input_column].astype(str),
                'reference': df[self.reference_column].astype(str)
            })

            # Remove rows with NaN values (including "nan" strings from astype conversion)
            self._data = self._data.dropna()
            self._data = self._data[~self._data['input'].str.contains('nan', case=False, na=False)]
            self._data = self._data[~self._data['reference'].str.contains('nan', case=False, na=False)]

            logger.info(f"Loaded {len(self._data)} samples from CSV: {self.data_path}")

        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            raise

    def _load_data_from_annotation(self) -> None:
        """Load data from annotation file."""
        try:
            data = pd.read_csv(self.data_path)
            data["input"] = data.apply(lambda x: json.dumps({"Query": x["new_instruction"], "Answer": x["original_generation"]}), axis=1)
            data["reference"] = data.apply(lambda x: json.dumps({
                "subject_score": x["adherence_subject"], 
                "exercise_type_score": x["adherence_exercise_type"], 
                "grade_level_score": x["adherence_grade_level"], 
                "modifier_score": x["adherence_modifier"]
            }), axis=1)
            self._data = data[["input", "reference"]]
        except Exception as e:
            logger.error(f"Error loading annotation dataset: {e}")
            raise

    @property
    def datasize(self) -> int:
        """Get the total number of samples in the dataset."""
        if self._data is None:
            return 0
        return len(self._data)

    @property
    def batchsize(self) -> int:
        """Get the batch size."""
        return self.batch_size

    def batch_iter(self) -> Iterator[Tuple[List[str], List[str]]]:
        """
        Iterate over batches of (inputs, references).
        
        Yields:
            Tuple of (inputs, references) for each batch.
        """
        if self._data is None:
            return

        data = self._data.copy()
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size]
            inputs = batch['input'].tolist()
            references = batch['reference'].tolist()
            yield inputs, references

    def get_all_data(self) -> Tuple[List[str], List[str]]:
        """
        Get all data as lists.
        
        Returns:
            Tuple of (all_inputs, all_references).
        """
        if self._data is None:
            return [], []
        return self._data['input'].tolist(), self._data['reference'].tolist() 


SYSTEM_PROMPT = """
Role:
You are an exercise generation evaluator. 

Task:
Given an query-answer pair, where the query is a demand and the answer is the generated exercise,
you will score the adherence of the exercise from 4 perspectives:
- Subject: Exercise subject, can be math, reading, writing, etc. Score 1 if the subject is correct else 0.
- Exercise type: Define exercise type/format, e.g. multiple choice, fill in the blank, etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
- Grade level: Education level of the students in American school system 0-12. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
- Modifier: Additional instruction for the exercise for specific requirements, e.g. "It should question about algebra", etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.

Note: If the query does not mention any of the subject, exercise type, grade level, or modifier, set the score to null.
    For example:
    - Query: "Generate a fill in the blank math exercise for grade 3 students, "
    - Answer: "1 + 1 = __"
    - return: {
        "subject_score": 1,
        "exercise_type_score": 4,
        "grade_level_score": 1,
        "modifier_score": null
    }

    - Query: "'Give me a multiple choice for a 1st grade student learning arts. It should be an exercise on symmetrical shapes.'"
    - Answer: "'Draw 2 shapes that are asymmetrical"
    - return: {
        "subject_score": 1,
        "exercise_type_score": 0,
        "grade_level_score": 1,
        "modifier_score": 3
    }

    - Query: "Give me a short answer exercise for a student learning computer science. It should have multiple steps. It should be an exercise on class diagrams."
    - Answer: "Who proposed the theory of evolution?"
    - return: {
        "subject_score": 0,
        "exercise_type_score": 0,
        "grade_level_score": null,
        "modifier_score": 0
    }

Return your response using the structured format provided. For each score, please explain your reasoning clearly, detailing how the exercise aligns or misaligns with the criteria.
"""

def main(save_scores: bool = False):
    dataset = Dataset(
        data_path="merged_annotation_dataset.csv",
        batch_size=100,
        load_from_annotation=True,
    )
    generator = OpenAIGenerator(model="gpt-4.1")
    evaluator = AdherenceMatchEvaluator()
    generation_kwargs = {
        "system_prompt": SYSTEM_PROMPT,
        "response_format": _AdherenceSchema,
        "max_tokens": 1024,
        "temperature": 0.1
    }
    generated_outputs: List[str] = []
    all_inputs: List[str] = []
    all_references: List[str] = []

    for i, (inputs, references) in enumerate(dataset.batch_iter()):
        logger.info(f"Processing batch {i}")
        batch_outputs = generator.run_batch(inputs, **generation_kwargs)

        # Accumulate batch results
        generated_outputs.extend(batch_outputs)
        all_inputs.extend(inputs)
        all_references.extend(references)

        logger.info(f"Generated {len(batch_outputs)} outputs")

    # Use the *full* set of inputs / references when computing the final score
    score, individual_scores = evaluator.run_detailed(all_inputs, all_references, generated_outputs)
    logger.info(f"Score: {score}")

    # Save generated outputs to csv
    if save_scores:
        eval_result = pd.DataFrame(generated_outputs, columns=["generations"])
        eval_result["match_scores"] = individual_scores
        eval_result["inputs"] = all_inputs
        eval_result["references"] = all_references
        eval_result.to_csv("match_scores.csv", index=False)



if __name__ == "__main__":
    main(save_scores=True)
