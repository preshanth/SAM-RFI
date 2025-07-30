"""
SAM Adapter Registry System

Registry for managing SAM version adapters without "factory" terminology.
Provides clean way to register and retrieve different SAM implementations.
"""

from typing import Dict, Type, Optional, List, Any
import logging
from .base import SAMAdapter

logger = logging.getLogger(__name__)


class SAMRegistry:
    """Registry for managing SAM version adapters"""
    
    _adapters: Dict[str, Type[SAMAdapter]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, 
                 version: str, 
                 adapter_class: Type[SAMAdapter],
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a SAM adapter for a specific version
        
        Args:
            version: SAM version identifier (e.g., 'sam2', 'sam2_large')
            adapter_class: The adapter class to register
            metadata: Optional metadata about the adapter
        """
        if not issubclass(adapter_class, SAMAdapter):
            raise TypeError(f"Adapter class must inherit from SAMAdapter, got {adapter_class}")
        
        cls._adapters[version] = adapter_class
        cls._metadata[version] = metadata or {}
        
        logger.info(f"Registered SAM adapter: {version}")
    
    @classmethod
    def get_adapter(cls, version: str, **kwargs) -> SAMAdapter:
        """
        Get SAM adapter instance for specified version
        
        Args:
            version: SAM version identifier
            **kwargs: Arguments to pass to adapter constructor
            
        Returns:
            Initialized SAM adapter instance
        """
        if version not in cls._adapters:
            available = list(cls._adapters.keys())
            raise ValueError(f"Unknown SAM version: {version}. Available: {available}")
        
        adapter_class = cls._adapters[version]
        
        try:
            return adapter_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create adapter for {version}: {e}")
            raise
    
    @classmethod
    def list_versions(cls) -> List[str]:
        """List available SAM versions"""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_metadata(cls, version: str) -> Dict[str, Any]:
        """Get metadata for a specific SAM version"""
        if version not in cls._metadata:
            raise ValueError(f"Unknown SAM version: {version}")
        return cls._metadata[version].copy()
    
    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered adapters"""
        info = {}
        for version in cls._adapters:
            info[version] = {
                'class': cls._adapters[version].__name__,
                'metadata': cls._metadata.get(version, {}),
                'available': True
            }
        return info
    
    @classmethod
    def is_available(cls, version: str) -> bool:
        """Check if a SAM version is available"""
        return version in cls._adapters
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (mainly for testing)"""
        cls._adapters.clear()
        cls._metadata.clear()
        logger.info("Cleared all registered SAM adapters")
    
    @classmethod
    def get_recommended_version(cls, 
                               memory_gb: int = 16,
                               time_limit_hours: Optional[float] = None) -> str:
        """
        Get recommended SAM version based on hardware constraints
        
        Args:
            memory_gb: Available GPU memory in GB
            time_limit_hours: Time limit for operations (if any)
            
        Returns:
            Recommended SAM version identifier
        """
        available = cls.list_versions()
        
        if not available:
            raise RuntimeError("No SAM adapters registered")
        
        # Simple recommendation logic
        if memory_gb <= 16:
            # V100 or similar - prefer smaller models
            for version in ['sam2_small', 'sam2_tiny', 'sam2']:
                if version in available:
                    return version
        else:
            # H200 or similar - can handle larger models
            for version in ['sam2_large', 'sam2', 'sam2_small']:
                if version in available:
                    return version
        
        # Fallback to first available
        return available[0]


# Global registry instance
sam_registry = SAMRegistry()


def register_sam_adapter(version: str, 
                        adapter_class: Type[SAMAdapter],
                        metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenient function to register a SAM adapter"""
    sam_registry.register(version, adapter_class, metadata)


def get_sam_adapter(version: str, **kwargs) -> SAMAdapter:
    """Convenient function to get a SAM adapter"""
    return sam_registry.get_adapter(version, **kwargs)


def list_sam_versions() -> List[str]:
    """Convenient function to list available SAM versions"""
    return sam_registry.list_versions()