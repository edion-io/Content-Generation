import json
import os
import re
from openai import OpenAI, AsyncOpenAI
from typing import Any, Dict, Literal, Optional, List
import asyncio
import logging

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)


logger = logging.getLogger(__name__)


class OpenAIGenerator:
    """Generator using OpenAI API or compatible APIs."""

    def __init__(self, model: str = "gpt-4o-mini", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7,
                 system_prompt: str = None):
        """
        Initialize OpenAI generator.
        
        Args:
            model: Model name to use.
            api_key: OpenAI API key (defaults to env var OPENAI_API_KEY).
            base_url: Base URL for OpenAI compatible APIs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            system_prompt: System prompt to use.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

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
                            "content": kwargs.get("system_prompt", self.system_prompt),
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
                    temperature=kwargs.get('temperature', self.temperature)
                )
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
                            "content": kwargs.get("system_prompt", self.system_prompt),
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
    
    @classmethod
    def parse_response(cls, response: str) -> Optional[Dict[str, Any]]:
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

            # If the response format is AdherenceWithExplanationSchema, convert it to a dictionary
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
        

class BinaryAdherenceScore(BaseModel):
    score: Literal[0, 1] = Field(description="The adherence score for binary adherence")
    explanation: str = Field(description="The explanation for the adherence score")

class OrdinalAdherenceScore(BaseModel):
    score: Literal[0, 1, 2, 3, 4] = Field(description="The adherence score for ordinal adherence")
    explanation: str = Field(description="The explanation for the adherence score")

class AdherenceWithExplanationSchema(BaseModel):
    """Structured schema for adherence LLM output.
    This model is used to enforce JSON mode responses when interacting
    with the generator. All fields are optional because the query may
    legitimately omit certain criteria (the correct value in that case
    is ``null`` / ``None``).
    """
    subject: BinaryAdherenceScore = Field(description="The adherence score for the subject")
    exercise_type: OrdinalAdherenceScore = Field(description="The adherence score for the exercise type")
    grade_level: OrdinalAdherenceScore = Field(description="The adherence score for the grade level")
    modifier: OrdinalAdherenceScore = Field(description="The adherence score for the modifier")

class AdherenceSchema(BaseModel):
    """Structured schema for adherence LLM output.
    This model is used to enforce JSON mode responses when interacting
    with the generator. Fields that are not provided will default to None.
    """

    subject_score: Literal[0, 1] = Field(..., description="The adherence score for the subject, binary")
    exercise_type_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the exercise type, ordinal")
    grade_level_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the grade level, ordinal")
    modifier_score: Optional[Literal[0, 1, 2, 3, 4]] = Field(None, description="The adherence score for the modifier, ordinal")