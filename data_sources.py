"""
Data source management and retrieval.

This module handles loading data from various sources
and preparing it for validation and delivery to the model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from context_validator import ContextRecord


class DataSourceError(Exception):
    """Raised when a data source cannot be accessed or parsed."""
    pass


class DataSource:
    """Base class for data sources."""
    
    def __init__(self, name: str, authoritative: bool = False):
        self.name = name
        self.authoritative = authoritative
    
    def fetch(self) -> Optional[ContextRecord]:
        """Fetch data from this source. Must be implemented by subclasses."""
        raise NotImplementedError


class JSONFileSource(DataSource):
    """Data source that reads from a JSON file."""
    
    def __init__(self, name: str, file_path: str, authoritative: bool = False):
        super().__init__(name, authoritative)
        self.file_path = Path(file_path)
    
    def fetch(self) -> Optional[ContextRecord]:
        """
        Fetch data from a JSON file.
        
        Expected JSON format:
        {
            "value": "string content",
            "updated_at": "ISO 8601 timestamp"
        }
        
        Returns:
            A ContextRecord if the file exists and is valid, None otherwise
            
        Raises:
            DataSourceError: If the file exists but cannot be parsed
        """
        if not self.file_path.exists():
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            updated_at = datetime.fromisoformat(data['updated_at'])
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            
            return ContextRecord(
                source=self.name,
                updated_at=updated_at,
                value=data['value'],
                authoritative=self.authoritative
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise DataSourceError(
                f"Failed to parse {self.file_path}: {str(e)}"
            )


class DataSourceRegistry:
    """Registry for managing multiple data sources."""
    
    def __init__(self):
        self.sources = {}
    
    def register(self, source: DataSource):
        """Register a data source."""
        self.sources[source.name] = source
    
    def fetch(self, source_name: str) -> Optional[ContextRecord]:
        """
        Fetch data from a registered source.
        
        Args:
            source_name: Name of the registered source
            
        Returns:
            A ContextRecord if available, None if source not found or unavailable
            
        Raises:
            DataSourceError: If the source fails during fetching
        """
        if source_name not in self.sources:
            return None
        
        return self.sources[source_name].fetch()
    
    def fetch_first_available(self, source_names: list) -> Optional[ContextRecord]:
        """
        Fetch from the first available source in the list.
        
        Tries sources in order and returns the first successful result.
        This implements a fallback strategy while respecting source priority.
        
        Args:
            source_names: List of source names to try in order
            
        Returns:
            A ContextRecord from the first available source, or None
        """
        for source_name in source_names:
            try:
                record = self.fetch(source_name)
                if record is not None:
                    return record
            except DataSourceError:
                # Try next source
                continue
        
        return None


def create_default_registry() -> DataSourceRegistry:
    """
    Create and populate the default data source registry.
    
    Sets up sources from sample_data directory.
    
    Returns:
        A populated DataSourceRegistry
    """
    registry = DataSourceRegistry()
    
    # Live feed is authoritative
    registry.register(JSONFileSource(
        name="enterprise_operational_feed",
        file_path="sample_data/live_feed.json",
        authoritative=True
    ))
    
    # Manual data is not authoritative
    registry.register(JSONFileSource(
        name="manual_business_entry",
        file_path="sample_data/stale_manual_data.json",
        authoritative=False
    ))
    
    return registry