"""
Model cache management for SAM-RFI

Handles downloading and caching SAM3 models from HuggingFace.
Provides progress bars and cache location management.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

try:
    from transformers import Sam3Model, Sam3Processor
    from huggingface_hub import snapshot_download, hf_hub_download
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Install with: pip install transformers huggingface_hub tqdm"
    )


class ModelCache:
    """
    Manage SAM3 model downloads and caching.

    SAM3 models are automatically downloaded from HuggingFace and cached locally.
    Default cache location: ~/.cache/huggingface/hub/

    Available model:
    - unified: facebook/sam3 (~3440MB / 3.4GB, 840M parameters)

    Note: Unlike SAM2, SAM3 has a single unified model with no size variants.

    Example:
        >>> from samrfi.utils import ModelCache
        >>>
        >>> # Check if model is cached
        >>> cache = ModelCache()
        >>> is_cached = cache.is_cached('unified')
        >>>
        >>> # Get cache info
        >>> info = cache.get_cache_info('unified')
        >>> print(f"Model size: {info['size_mb']:.1f} MB")
        >>>
        >>> # Pre-download model with progress bar
        >>> cache.download_model('unified', show_progress=True)
        >>>
        >>> # Load model (auto-downloads if not cached)
        >>> model, processor = cache.load_model('unified')
    """

    # Map checkpoint names to HuggingFace model IDs
    # SAM3 has single unified model (840M params)
    CHECKPOINT_MAP = {
        "unified": "facebook/sam3",
    }

    # Approximate model sizes (in MB)
    MODEL_SIZES = {
        "unified": 3440,  # 3.44 GB
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ModelCache.

        Args:
            cache_dir: Optional custom cache directory. If None, uses HuggingFace default
                      (~/.cache/huggingface/hub/)
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir

    def get_model_id(self, checkpoint: str) -> str:
        """
        Get HuggingFace model ID for checkpoint name.

        Args:
            checkpoint: Checkpoint name (tiny, small, base_plus, large)

        Returns:
            HuggingFace model ID (e.g., 'facebook/sam2-hiera-large')

        Raises:
            ValueError: If checkpoint name is invalid
        """
        if checkpoint not in self.CHECKPOINT_MAP:
            valid = ', '.join(self.CHECKPOINT_MAP.keys())
            raise ValueError(
                f"Invalid checkpoint '{checkpoint}'. "
                f"Valid options: {valid}"
            )
        return self.CHECKPOINT_MAP[checkpoint]

    def is_cached(self, checkpoint: str) -> bool:
        """
        Check if model is already cached locally.

        Args:
            checkpoint: Checkpoint name (tiny, small, base_plus, large)

        Returns:
            True if model is cached, False otherwise
        """
        model_id = self.get_model_id(checkpoint)

        try:
            # Try to load from cache without downloading
            from huggingface_hub import try_to_load_from_cache
            from transformers import cached_file

            # Check if config.json exists in cache
            config_path = cached_file(
                model_id,
                "config.json",
                cache_dir=self.cache_dir,
                local_files_only=True,
                _raise_exceptions_for_missing_entries=False
            )
            return config_path is not None
        except Exception:
            return False

    def get_cache_info(self, checkpoint: str) -> dict:
        """
        Get cache information for a model.

        Args:
            checkpoint: Checkpoint name (tiny, small, base_plus, large)

        Returns:
            Dictionary with cache info:
                - is_cached: bool
                - model_id: str
                - size_mb: float (approximate)
                - cache_path: str (if cached)
        """
        model_id = self.get_model_id(checkpoint)
        is_cached = self.is_cached(checkpoint)

        info = {
            'is_cached': is_cached,
            'model_id': model_id,
            'size_mb': self.MODEL_SIZES.get(checkpoint, 0),
        }

        if is_cached:
            # Try to find cache path
            try:
                from transformers import cached_file
                config_path = cached_file(
                    model_id,
                    "config.json",
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                if config_path:
                    info['cache_path'] = str(Path(config_path).parent)
            except Exception:
                pass

        return info

    def download_model(
        self,
        checkpoint: str,
        show_progress: bool = True,
        force_download: bool = False
    ) -> str:
        """
        Download model from HuggingFace (if not cached).

        Args:
            checkpoint: Checkpoint name (tiny, small, base_plus, large)
            show_progress: Show download progress bar
            force_download: Force re-download even if cached

        Returns:
            Path to cached model directory
        """
        model_id = self.get_model_id(checkpoint)

        if not force_download and self.is_cached(checkpoint):
            info = self.get_cache_info(checkpoint)
            cache_path = info.get('cache_path', 'unknown')
            if show_progress:
                print(f"✓ Model '{checkpoint}' already cached at: {cache_path}")
            return cache_path

        if show_progress:
            size_mb = self.MODEL_SIZES.get(checkpoint, 0)
            print(f"Downloading SAM2 model '{checkpoint}' (~{size_mb} MB)...")
            print(f"Model ID: {model_id}")
            print(f"Cache: {self.cache_dir or '~/.cache/huggingface/hub/'}")

        # Download entire model with progress
        cache_path = snapshot_download(
            model_id,
            cache_dir=self.cache_dir,
            resume_download=True,
            force_download=force_download,
            tqdm_class=tqdm if show_progress else None
        )

        if show_progress:
            print(f"✓ Download complete: {cache_path}")

        return cache_path

    def load_model(
        self,
        checkpoint: str,
        show_progress: bool = True,
        device: str = "cuda"
    ) -> Tuple[Sam3Model, Sam3Processor]:
        """
        Load SAM3 model and processor (auto-downloads if not cached).

        Args:
            checkpoint: Checkpoint name (unified)
            show_progress: Show download progress if model not cached
            device: Device to load model on ('cuda' or 'cpu')

        Returns:
            Tuple of (model, processor)
        """
        model_id = self.get_model_id(checkpoint)

        # Check cache status
        is_cached = self.is_cached(checkpoint)

        if not is_cached and show_progress:
            size_mb = self.MODEL_SIZES.get(checkpoint, 0)
            print(f"\nModel '{checkpoint}' not found in cache.")
            print(f"Downloading from HuggingFace (~{size_mb} MB)...")
            print(f"This is a one-time download. Subsequent runs will use cached model.\n")

        # Load model and processor (auto-downloads if needed)
        if show_progress and not is_cached:
            print(f"Loading SAM3 processor...")
        processor = Sam3Processor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        )

        if show_progress and not is_cached:
            print(f"Loading SAM3 model...")
        model = Sam3Model.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        )

        # Move to device
        model = model.to(device)

        if show_progress:
            info = self.get_cache_info(checkpoint)
            cache_path = info.get('cache_path', 'cache')
            print(f"✓ Model loaded: {model_id}")
            print(f"  Device: {device}")
            print(f"  Cache: {cache_path}\n")

        return model, processor

    def clear_cache(self, checkpoint: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            checkpoint: Checkpoint to clear. If None, prints cache info only.

        Warning:
            This deletes cached model files. They will need to be re-downloaded.
        """
        if checkpoint is None:
            print("Cache information:")
            for ckpt in self.CHECKPOINT_MAP.keys():
                info = self.get_cache_info(ckpt)
                status = "✓ cached" if info['is_cached'] else "✗ not cached"
                print(f"  {ckpt:12} ({info['size_mb']:4.0f} MB): {status}")
            print("\nTo clear a specific model: clear_cache('checkpoint_name')")
            return

        model_id = self.get_model_id(checkpoint)
        info = self.get_cache_info(checkpoint)

        if not info['is_cached']:
            print(f"Model '{checkpoint}' is not cached.")
            return

        cache_path = info.get('cache_path')
        if cache_path and os.path.exists(cache_path):
            import shutil
            shutil.rmtree(cache_path)
            print(f"✓ Cleared cache for '{checkpoint}': {cache_path}")
        else:
            print(f"Could not find cache path for '{checkpoint}'")

    @staticmethod
    def list_available_models() -> None:
        """Print list of available SAM3 models with sizes."""
        print("Available SAM3 model:")
        print("-" * 60)
        for checkpoint, model_id in ModelCache.CHECKPOINT_MAP.items():
            size_mb = ModelCache.MODEL_SIZES.get(checkpoint, 0)
            print(f"  {checkpoint:12} | {size_mb:4.0f} MB | {model_id}")
        print("-" * 60)
        print("Note: SAM3 has single unified model (840M params, no variants)")
        print("Usage: ModelCache().load_model('unified')")
