# AI Data Freshness Guardrail

A small production-minded AI example that demonstrates how to validate **data source, freshness, and provenance before information is provided to a large language model**.

This project was designed around a common enterprise AI failure mode: a model can produce a logically consistent answer while still being wrong because the context supplied to it is stale, manually maintained, or no longer authoritative.

The goal is not to build another chatbot. The goal is to demonstrate how to build **trustworthy AI context pipelines**.

---

## Why This Project Exists

In production AI systems, unexpected output is not always a model problem.

A model may behave correctly relative to the information it receives while still producing a business result that is outdated or incorrect.

For example:

- An automated enterprise feed contains current operational data.
- A manually maintained data source was originally used during development or testing.
- The manual source remains accessible to the application.
- The model receives the manual value instead of the current feed.
- The model produces a reasonable answer based on stale information.

The failure is not necessarily in the model.

It may be in the **data path**.

This project demonstrates an architectural pattern for preventing that class of issue.

---

## Core Principle

Before data is included in model context, validate:

1. **Source** — Is this an approved, authoritative source?
2. **Freshness** — Is the data recent enough for the use case?
3. **Provenance** — Can we identify where the model's context came from?
4. **Availability** — If trusted data is unavailable, should the AI request fail safely?
5. **Observability** — Can operators determine what data the model used?

The model should never have to decide whether a source is trustworthy.

That decision should happen **before the prompt is constructed**.

---

## Project Structure

```text
ai-data-guardrail/
│
├── README.md
├── requirements.txt
├── app.py
├── data_sources.py
├── context_validator.py
├── model_service.py
│
├── tests/
│   └── test_context_validator.py
│
└── sample_data/
    ├── live_feed.json
    └── stale_manual_data.json
```

---

## Architecture

```text
Enterprise Data Feed
        │
        ▼
   Data Source Layer
        │
        ▼
 Context Validation
 ┌───────────────────┐
 │ Approved source?  │
 │ Data current?     │
 │ Timestamp valid?  │
 │ Provenance known? │
 └───────────────────┘
        │
        ├── FAIL ──► Block AI request / alert
        │
        ▼
 Validated Context
        │
        ▼
   Prompt Builder
        │
        ▼
      LLM
        │
        ▼
 Response + Provenance
```

---

## Example Context Model

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ContextRecord:
    source: str
    updated_at: datetime
    value: str
    authoritative: bool = False
```

Each record includes the information needed to make a trust decision before the data reaches the model.

---

## Freshness and Source Validation

```python
from datetime import datetime, timedelta, timezone


class ContextValidationError(Exception):
    pass


def validate_context(
    record: ContextRecord,
    max_age_minutes: int = 30
) -> ContextRecord:
    """Validate data before it is provided to the AI model."""

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
```

This creates a hard boundary between trusted enterprise data and model context.

---

## Valid Data Example

```python
from datetime import datetime, timezone


feed_record = ContextRecord(
    source="enterprise_operational_feed",
    updated_at=datetime.now(timezone.utc),
    value="Current inventory level: 8,412 units",
    authoritative=True,
)

validated = validate_context(feed_record)
```

This record is accepted because:

- the source is approved,
- it is marked authoritative,
- and the data is current.

---

## Stale or Manual Data Example

```python
from datetime import datetime, timedelta, timezone


manual_record = ContextRecord(
    source="manual_business_entry",
    updated_at=datetime.now(timezone.utc) - timedelta(days=3),
    value="Current inventory level: 9,750 units",
    authoritative=False,
)

try:
    validate_context(manual_record)
except ContextValidationError as exc:
    print(f"AI request blocked: {exc}")
```

Expected output:

```text
AI request blocked: Rejected non-authoritative source: manual_business_entry
```

The key behavior is intentional:

**the request is blocked before stale or untrusted information can reach the model.**

---

## Building Model Context

Only validated information should be used to construct the prompt.

```python
def build_model_context(record: ContextRecord) -> str:
    validated = validate_context(record)

    return f"""
SOURCE: {validated.source}
LAST_UPDATED: {validated.updated_at.isoformat()}
DATA_STATUS: CURRENT_AND_VALIDATED

BUSINESS_DATA:
{validated.value}
""".strip()
```

This makes provenance part of the model context itself.

---

## Prompt Construction

```python
def create_prompt(question: str, context: str) -> str:
    return f"""
You are an enterprise analytics assistant.

Answer using only the validated business context below.

If the supplied data does not answer the question,
state that current authoritative data is unavailable.

{context}

QUESTION:
{question}
""".strip()
```

The model is instructed to operate only on validated context.

This reduces the likelihood of the application silently substituting untrusted or incomplete information.

---

## Returning Provenance

Model output should include enough metadata for the application or operator to understand the source of the answer.

```python
def build_response(answer: str, record: ContextRecord) -> dict:
    return {
        "answer": answer,
        "provenance": {
            "source": record.source,
            "updated_at": record.updated_at.isoformat(),
            "authoritative": record.authoritative,
        }
    }
```

Example response:

```json
{
  "answer": "The current inventory level is 8,412 units.",
  "provenance": {
    "source": "enterprise_operational_feed",
    "updated_at": "2026-10-01T12:00:00+00:00",
    "authoritative": true
  }
}
```

---

## Testing

The project includes automated tests for the most important controls.

```python
from datetime import datetime, timedelta, timezone

import pytest

from context_validator import (
    ContextRecord,
    ContextValidationError,
    validate_context,
)


def test_current_authoritative_data_is_accepted():
    record = ContextRecord(
        source="enterprise_feed",
        updated_at=datetime.now(timezone.utc),
        value="Current value",
        authoritative=True,
    )

    assert validate_context(record) == record


def test_manual_data_is_rejected():
    record = ContextRecord(
        source="manual_entry",
        updated_at=datetime.now(timezone.utc),
        value="Old value",
        authoritative=False,
    )

    with pytest.raises(ContextValidationError):
        validate_context(record)


def test_stale_feed_is_rejected():
    record = ContextRecord(
        source="enterprise_feed",
        updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
        value="Stale value",
        authoritative=True,
    )

    with pytest.raises(ContextValidationError):
        validate_context(record, max_age_minutes=30)
```

Run the tests with:

```bash
pytest
```

---

## Design Decisions

### Fail Closed

If authoritative data is unavailable, the safest behavior is to prevent the AI request rather than silently fall back to an unverified source.

### Validate Before Prompt Construction

Trust decisions should be deterministic application logic.

They should not be delegated to the LLM.

### Make Provenance Visible

Operators should be able to determine:

- what source supplied the data,
- when it was updated,
- whether it passed validation,
- and what information was ultimately presented to the model.

### Separate Model Quality From Data Quality

When an AI response appears incorrect, troubleshoot the entire chain:

```text
Model Output
    ↓
Prompt / Context
    ↓
Data Transformation
    ↓
Data Pipeline
    ↓
Source
    ↓
Freshness
```

A model can be functioning exactly as designed while consuming the wrong information.

---

## Production Extensions

A production implementation could extend this pattern with:

- schema validation,
- data-quality scoring,
- freshness SLAs by source,
- source allowlists,
- feed health monitoring,
- lineage metadata,
- OpenTelemetry tracing,
- structured logging,
- alerting for stale feeds,
- fallback policies,
- model and prompt versioning,
- response audit records,
- human approval for high-risk actions,
- MLflow experiment and model tracking,
- vector-store document timestamps,
- retrieval provenance,
- policy-based access controls.

---

## Example Enterprise Use Cases

This pattern can be applied to:

- inventory and supply-chain assistants,
- operations dashboards,
- forecasting systems,
- executive analytics assistants,
- customer-service AI,
- retrieval-augmented generation systems,
- manufacturing analytics,
- pricing systems,
- financial analysis,
- incident-response assistants,
- enterprise knowledge agents.

---

## What This Project Demonstrates

This project intentionally focuses on the engineering around the model rather than the novelty of making an API call to an LLM.

It demonstrates:

- production-minded AI architecture,
- data provenance,
- context validation,
- data freshness controls,
- defensive AI engineering,
- separation of concerns,
- testable guardrails,
- observability thinking,
- fail-safe behavior,
- enterprise AI operating principles.

---

## Key Takeaway

A useful production AI principle is:

> A model may be correct relative to the context it was given and still produce the wrong business outcome.

Reliable AI therefore requires more than model selection or prompt engineering.

It requires trustworthy data pipelines, authoritative sources, freshness validation, provenance, observability, and clear operating controls around the model.

---

## License

This project is intended as a reference implementation and demonstration of production AI architecture patterns.
