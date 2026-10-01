"""
Context validation for AI model safety.

This module ensures that data supplied to AI models meets strict
freshness, source, and provenance requirements before being used
in prompt construction.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


class ContextValidationError(Exception):
    """Raised when context validation fails."""
    pass


@dataclass
class ContextRecord:
    """Represents a single piece of context with metadata."""
    source: str
    updated_at: datetime
    value: str
    authoritative: bool = False


def validate_context(
    record: ContextRecord,
    max_age_minutes: int = 30
) -> ContextRecord:
    """
    Validate data before it is provided to the AI model.
    
    Ensures that:
    - The source is marked as authoritative
    - The data is recent enough for the use case
    
    Args:
        record: The context record to validate
        max_age_minutes: Maximum age in minutes for the data
        
    Returns:
        The validated context record
        
    Raises:
        ContextValidationError: If validation fails
    """
    if not record.authoritative:
        raise ContextValidationError(
            f"Rejected non-authoritative source: {record.source}"
        )

    now = datetime.now(timezone.utc)
    age = now - record.updated_at

    if age > timedelta(minutes=max_age_minutes):
        raise ContextValidationError(
            f"Rejected stale data from {record.source}. "
            f"Age: {age.total_seconds() / 60:.1f} minutes"
        )

    return record


def build_model_context(record: ContextRecord) -> str:
    """
    Build context string for the model prompt.
    
    Only validated information is used to construct the prompt.
    Provenance is embedded in the context itself.
    
    Args:
        record: The context record (must be validated first)
        
    Returns:
        A formatted context string for the model
        
    Raises:
        ContextValidationError: If the record fails validation
    """
    validated = validate_context(record)

    return f"""
SOURCE: {validated.source}
LAST_UPDATED: {validated.updated_at.isoformat()}
DATA_STATUS: CURRENT_AND_VALIDATED

BUSINESS_DATA:
{validated.value}
""".strip()


def create_prompt(question: str, context: str) -> str:
    """
    Create a prompt with validated context.
    
    The model is instructed to operate only on validated context
    and to fail safely if data is unavailable.
    
    Args:
        question: The user's question
        context: The validated business context
        
    Returns:
        The complete prompt for the model
    """
    return f"""
You are an enterprise analytics assistant.

Answer using only the validated business context below.

If the supplied data does not answer the question,
state that current authoritative data is unavailable.

{context}

QUESTION:
{question}
""".strip()


def build_response(answer: str, record: ContextRecord) -> dict:
    """
    Build response with provenance metadata.
    
    Ensures that the source and freshness of information
    are visible to the operator or downstream system.
    
    Args:
        answer: The model's answer
        record: The context record that was used
        
    Returns:
        A response dictionary with answer and provenance
    """
    return {
        "answer": answer,
        "provenance": {
            "source": record.source,
            "updated_at": record.updated_at.isoformat(),
            "authoritative": record.authoritative,
        }
    }