"""
Model service for AI interaction.

This module demonstrates how to safely interact with an LLM
while respecting data validation guardrails.
"""

from context_validator import (
    ContextRecord,
    ContextValidationError,
    build_model_context,
    create_prompt,
    build_response,
)
from data_sources import DataSourceRegistry


class ModelServiceError(Exception):
    """Raised when the model service cannot proceed."""
    pass


class ModelService:
    """Service for safely querying the model with validated context."""
    
    def __init__(self, registry: DataSourceRegistry):
        """
        Initialize the model service.
        
        Args:
            registry: DataSourceRegistry for loading context
        """
        self.registry = registry
    
    def query(
        self,
        question: str,
        source_names: list,
        max_age_minutes: int = 30
    ) -> dict:
        """
        Query the model with validated context.
        
        Implements a fail-closed pattern: if trusted data is unavailable,
        the request fails rather than silently falling back to untrusted sources.
        
        Args:
            question: The user's question
            source_names: List of data source names to try, in order of preference
            max_age_minutes: Maximum age in minutes for acceptable data
            
        Returns:
            A response dictionary with answer and provenance
            
        Raises:
            ModelServiceError: If data validation fails or no trusted data is available
        """
        record = self.registry.fetch_first_available(source_names)
        
        if record is None:
            raise ModelServiceError(
                f"No data available from sources: {source_names}"
            )
        
        try:
            context = build_model_context(record)
        except ContextValidationError as e:
            raise ModelServiceError(
                f"Data validation failed: {str(e)}. "
                f"AI request blocked to prevent stale or untrusted information."
            ) from e
        
        prompt = create_prompt(question, context)
        answer = self._generate_answer(prompt, record)
        return build_response(answer, record)
    
    def _generate_answer(self, prompt: str, record: ContextRecord) -> str:
        """
        Generate an answer from the model.
        
        In production, this would call an LLM API (OpenAI, Anthropic, etc.).
        For demonstration, we return a synthesized answer based on the context.
        
        Args:
            prompt: The complete prompt to send to the model
            record: The validated context record
            
        Returns:
            The model's answer
        """
        return f"Based on the current operational data, {record.value.lower()}"