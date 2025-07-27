import os
from dotenv import load_dotenv
from typing import List

import numpy as np
from eval.adherence.generator import AdherenceSchema, OpenAIGenerator

# Set up your OpenAI API key
load_dotenv()
assert os.getenv(
    "OPENAI_API_KEY"
), "OpenAI API key not found, please set it in environment variable OPENAI_API_KEY or in .env file"


# Prompt engineering
SYSTEM_PROMPT = """
Role:
You are an exercise generation evaluator. 

Task:
Given an query-answer pair, where the query is a demand and the answer is the generated exercise,
you will score the adherence of the exercise from 4 perspectives:
- Subject: Exercise subject, can be math, science, social science, history, English, French, German, biology, art, etc. Score 1 if the subject is correct else 0.
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

Return your response using the structured format provided.
{"subject_score": x, "exercise_type_score": x, "grade_level_score": x, "modifier_score": x}
"""


class AdherenceEvaluator:

    def __init__(
        self,
        generator_model: str = "gpt-4o-mini",
        system_prompt: str = None,
        subject_weight: float = 0.25,
        exercise_type_weight: float = 0.25,
        grade_level_weight: float = 0.25,
        modifier_weight: float = 0.25,
    ):
        self.generator_model = generator_model
        self.system_prompt = system_prompt
        self.subject_weight = subject_weight
        self.exercise_type_weight = exercise_type_weight
        self.grade_level_weight = grade_level_weight
        self.modifier_weight = modifier_weight
        self.generator = OpenAIGenerator(model="gpt-4o-mini")
        self.generation_kwargs = {
            "system_prompt": system_prompt or SYSTEM_PROMPT,
            "response_format": AdherenceSchema,
            "max_tokens": 1024,
            "temperature": 0.1
        }

    def calculate_overall_score(self, result: dict, mask: List[int] | np.ndarray = [0, 0, 0, 0]):
        """
        Calculate the overall score for the result.

        Args:
            result: Dict, the result of the adherence score.
            mask: List[int] | np.ndarray, the mask of the result. 1 means the corresponding value should be ignored
        """
        sub_scores = {}
        active_weights = {}

        # Define weight and mask mappings
        weight_mask_mappings = {
            "subject_score": (self.subject_weight, mask[0]),
            "exercise_type_score": (self.exercise_type_weight, mask[1]),
            "grade_level_score": (self.grade_level_weight, mask[2]),
            "modifier_score": (self.modifier_weight, mask[3])
        }

        for field, (weight_value, mask_value) in weight_mask_mappings.items():
            sub_score = result.get(field)

            if sub_score is None or mask_value == 1:
                # If sub_score is None, exclude from scoring
                continue
            else:
                if field == "subject_score":
                    # Binary score
                    sub_scores[field] = sub_score
                else:
                    # Ordinal score is normalized to 0-1
                    sub_scores[field] = sub_score / 4

                active_weights[field] = weight_value

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
            normalized_weights[field] * sub_scores[field]
            for field in sub_scores
        )

        return max(0.0, min(1.0, total_score))

    def adherence_score(self, instructions: List[str], generations: List[str], masks: List[List[int]]|np.ndarray = None, **generation_kwargs):
        """
        Estimate adherence score for generated content in 4 aspects:
        - Subject: Exercise subject, can be math, reading, writing, etc. Score 1 if the subject is correct else 0.
        - Exercise type: Define exercise type/format, e.g. multiple choice, fill in the blank, etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
        - Grade level: Education level of the students in American school system 0-12. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
        - Modifier: Additional instruction for the exercise for specific requirements, e.g. "It should question about algebra", etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.

        Args:
            instructions: List[str], user prompts, the requests for exercise generation.
            generations: List[str], the generated exercises as the responses to instructions.
            masks: List[List[int]]|np.ndarray, the mask of the result, 1 means the corresponding value should be ignored
        """
        user_prompts = [
            str({"Query": instruction, "Answer": generation})
            for instruction, generation in zip(instructions, generations)
        ]
        responses = self.generator.run_batch(user_prompts, **self.generation_kwargs)
        try:
            if responses:
                results = [self.generator.parse_response(response) for response in responses]
                for i, result in enumerate(results):
                    if masks is None:
                        mask = [1, 1, 1, 1]
                    else:
                        mask = masks[i]
                    result["overall_score"] = self.calculate_overall_score(result, mask)
                return results
        except Exception as e:
            print(f"Error when parsing adherence score: {e}")
            return None


if __name__ == "__main__":
    evaluator = AdherenceEvaluator()
    instructions = [
        "Generate a math exercise for grade 3 students", 
        "Generate a fill in the blank math exercise for grade 5 students, it should be an exercise about addition",
        "Create a language arts exercise about adjectives for 4th grade.",
    ]
    generations = [
        "Who is the president of the United States?",
        "1 + 1 = __",
        "How many chemical elements?"
    ]
    masks = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    result = evaluator.adherence_score(
        instructions, generations, masks
    )
    print(result)

