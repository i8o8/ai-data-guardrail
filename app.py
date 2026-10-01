"""
AI Data Freshness Guardrail Application

A demonstration application that shows how to safely integrate
data validation guardrails into an AI query pipeline.
"""

from data_sources import create_default_registry
from model_service import ModelService, ModelServiceError


def main():
    """Run the demonstration application."""
    
    # Initialize data sources and model service
    registry = create_default_registry()
    service = ModelService(registry)
    
    print("=" * 70)
    print("AI Data Freshness Guardrail - Demonstration")
    print("=" * 70)
    print()
    
    # Test Case 1: Query with valid authoritative data
    print("Test Case 1: Query with valid authoritative data")
    print("-" * 70)
    try:
        response = service.query(
            question="What is the current inventory level?",
            source_names=["enterprise_operational_feed"]
        )
        print(f"✓ Request succeeded")
        print(f"  Answer: {response['answer']}")
        print(f"  Source: {response['provenance']['source']}")
        print(f"  Authoritative: {response['provenance']['authoritative']}")
        print(f"  Updated: {response['provenance']['updated_at']}")
    except ModelServiceError as e:
        print(f"✗ Request failed: {e}")
    print()
    
    # Test Case 2: Query with non-authoritative data
    print("Test Case 2: Query with non-authoritative data")
    print("-" * 70)
    try:
        response = service.query(
            question="What is the current inventory level?",
            source_names=["manual_business_entry"]
        )
        print(f"✓ Request succeeded")
        print(f"  Answer: {response['answer']}")
    except ModelServiceError as e:
        print(f"✗ Request blocked: {e}")
        print(f"  (This is the expected and desired behavior)")
    print()
    
    # Test Case 3: Fallback strategy
    print("Test Case 3: Fallback strategy (authoritative first, then manual)")
    print("-" * 70)
    try:
        response = service.query(
            question="What is the current inventory level?",
            source_names=[
                "enterprise_operational_feed",  # Try authoritative first
                "manual_business_entry"          # Manual as fallback
            ]
        )
        print(f"✓ Request succeeded")
        print(f"  Answer: {response['answer']}")
        print(f"  Source: {response['provenance']['source']}")
        print(f"  Authoritative: {response['provenance']['authoritative']}")
    except ModelServiceError as e:
        print(f"✗ Request failed: {e}")
    print()
    
    # Test Case 4: No data available
    print("Test Case 4: No data available (non-existent source)")
    print("-" * 70)
    try:
        response = service.query(
            question="What is the current inventory level?",
            source_names=["nonexistent_source"]
        )
        print(f"✓ Request succeeded")
        print(f"  Answer: {response['answer']}")
    except ModelServiceError as e:
        print(f"✗ Request failed: {e}")
        print(f"  (This is the expected and desired behavior)")
    print()
    
    print("=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
