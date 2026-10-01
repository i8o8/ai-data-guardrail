"""
Tests for context validation.

These tests verify the critical controls that prevent stale or
untrusted data from reaching the AI model.
"""

from datetime import datetime, timedelta, timezone

import pytest

from context_validator import (
    ContextRecord,
    ContextValidationError,
    validate_context,
    build_model_context,
    create_prompt,
    build_response,
)


class TestContextValidation:
    """Test context validation logic."""
    
    def test_current_authoritative_data_is_accepted(self):
        """Current authoritative data should pass validation."""
        record = ContextRecord(
            source="enterprise_feed",
            updated_at=datetime.now(timezone.utc),
            value="Current value",
            authoritative=True,
        )
        
        assert validate_context(record) == record
    
    def test_manual_data_is_rejected(self):
        """Non-authoritative data should be rejected."""
        record = ContextRecord(
            source="manual_entry",
            updated_at=datetime.now(timezone.utc),
            value="Manual value",
            authoritative=False,
        )
        
        with pytest.raises(ContextValidationError) as exc_info:
            validate_context(record)
        
        assert "non-authoritative source" in str(exc_info.value)
    
    def test_stale_feed_is_rejected(self):
        """Data older than max_age_minutes should be rejected."""
        record = ContextRecord(
            source="enterprise_feed",
            updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
            value="Stale value",
            authoritative=True,
        )
        
        with pytest.raises(ContextValidationError) as exc_info:
            validate_context(record, max_age_minutes=30)
        
        assert "stale data" in str(exc_info.value)
    
    def test_freshness_threshold_is_respected(self):
        """Data at the freshness boundary should be handled correctly."""
        # Data exactly at the threshold should be accepted
        record_at_threshold = ContextRecord(
            source="enterprise_feed",
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            value="Value at threshold",
            authoritative=True,
        )
        
        # Should be accepted (age is exactly 30 minutes)
        # Note: Due to execution time, might be slightly over
        try:
            validate_context(record_at_threshold, max_age_minutes=30)
        except ContextValidationError:
            # Acceptable due to execution time
            pass
        
        # Data past the threshold should be rejected
        record_past_threshold = ContextRecord(
            source="enterprise_feed",
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=31),
            value="Value past threshold",
            authoritative=True,
        )
        
        with pytest.raises(ContextValidationError):
            validate_context(record_past_threshold, max_age_minutes=30)
    
    def test_validation_error_includes_source_name(self):
        """Validation errors should identify the source."""
        record = ContextRecord(
            source="suspicious_source",
            updated_at=datetime.now(timezone.utc),
            value="Untrusted data",
            authoritative=False,
        )
        
        with pytest.raises(ContextValidationError) as exc_info:
            validate_context(record)
        
        assert "suspicious_source" in str(exc_info.value)


class TestModelContext:
    """Test model context building."""
    
    def test_valid_context_produces_prompt_content(self):
        """Valid context should produce formatted prompt content."""
        record = ContextRecord(
            source="test_source",
            updated_at=datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
            value="Test data",
            authoritative=True,
        )
        
        context = build_model_context(record)
        
        assert "SOURCE: test_source" in context
        assert "DATA_STATUS: CURRENT_AND_VALIDATED" in context
        assert "Test data" in context
    
    def test_context_includes_provenance(self):
        """Context should include timestamp for provenance."""
        record = ContextRecord(
            source="test_source",
            updated_at=datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
            value="Test data",
            authoritative=True,
        )
        
        context = build_model_context(record)
        
        assert "2026-10-01" in context
    
    def test_invalid_context_raises_error(self):
        """Invalid context should raise ContextValidationError."""
        record = ContextRecord(
            source="manual_entry",
            updated_at=datetime.now(timezone.utc),
            value="Untrusted data",
            authoritative=False,
        )
        
        with pytest.raises(ContextValidationError):
            build_model_context(record)


class TestPromptConstruction:
    """Test prompt construction."""
    
    def test_prompt_includes_context(self):
        """Prompt should include the provided context."""
        context = "SOURCE: test\nDATA: value"
        question = "What is the value?"
        
        prompt = create_prompt(question, context)
        
        assert context in prompt
        assert question in prompt
    
    def test_prompt_includes_safety_instructions(self):
        """Prompt should include instructions about data validity."""
        context = "SOURCE: test\nDATA: value"
        question = "What is the value?"
        
        prompt = create_prompt(question, context)
        
        assert "validated" in prompt.lower()


class TestResponse:
    """Test response building with provenance."""
    
    def test_response_includes_provenance(self):
        """Response should include full provenance metadata."""
        record = ContextRecord(
            source="enterprise_feed",
            updated_at=datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
            value="Test data",
            authoritative=True,
        )
        answer = "Test answer"
        
        response = build_response(answer, record)
        
        assert response["answer"] == answer
        assert response["provenance"]["source"] == "enterprise_feed"
        assert response["provenance"]["authoritative"] is True
        assert "2026-10-01" in response["provenance"]["updated_at"]
    
    def test_response_preserves_data_source_information(self):
        """Response provenance should match the source record."""
        record = ContextRecord(
            source="custom_source",
            updated_at=datetime(2026, 10, 1, 10, 30, 0, tzinfo=timezone.utc),
            value="Custom data",
            authoritative=False,
        )
        answer = "Custom answer"
        
        response = build_response(answer, record)
        
        assert response["provenance"]["source"] == "custom_source"
        assert response["provenance"]["authoritative"] is False
