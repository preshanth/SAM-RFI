"""
Model cache management for SAM-RFI.

This module handles automatic downloading and caching of SAM2 models from
HuggingFace Hub. Models are downloaded once and cached locally for subsequent
use, with configurable cache locations and progress tracking.

SAM2 models are downloaded from the facebook/sam2-hiera-* repositories on
HuggingFace and cached in ~/.cache/huggingface/hub/ by default.

Classes
-------
ModelCache
    Manages SAM2 model downloads, caching, and loading from HuggingFace.

Examples
--------
>>> from samrfi.utils.model_cache import ModelCache
>>>
>>> # Initialize cache manager
>>> cache = ModelCache()
>>>
>>> # Check if model is cached
>>> is_cached = cache.is_cached('large')
>>> print(f"Model cached: {is_cached}")
>>>
>>> # Get cache information
>>> info = cache.get_cache_info('large')
>>> print(f"Size: {info['size_mb']} MB")
>>> print(f"Cached: {info['is_cached']}")
>>>
>>> # Download model (with progress bar)
>>> cache.download_model('large', show_progress=True)
>>>
>>> # Load model and processor
>>> model, processor = cache.load_model('large', device='cuda')
>>>
>>> # List available models
>>> ModelCache.list_available_models()
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import Sam2Model, Sam2Processor
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Install with: pip install transformers huggingface_hub tqdm"
    ) from e


class ModelCache:
    """
    Manage SAM2 model downloads and caching.

    Handles automatic downloading of SAM2 models from HuggingFace Hub, local
    caching, and loading of models and processors. Models are downloaded once
    and reused from cache on subsequent calls.

    Default cache location is ~/.cache/huggingface/hub/, but can be customized
    via the cache_dir parameter or HF_HOME environment variable.

    Available SAM2 models:
    - tiny: facebook/sam2-hiera-tiny (~40MB, fastest inference)
    - small: facebook/sam2-hiera-small (~180MB, good speed/accuracy balance)
    - base_plus: facebook/sam2-hiera-base-plus (~330MB, higher accuracy)
    - large: facebook/sam2-hiera-large (~850MB, best accuracy)

    Parameters
    ----------
    cache_dir : str or None, optional
        Custom cache directory path. If None, uses HuggingFace default
        (~/.cache/huggingface/hub/). Default is None.

    Attributes
    ----------
    CHECKPOINT_MAP : dict
        Maps checkpoint names to HuggingFace model IDs.
    MODEL_SIZES : dict
        Approximate model sizes in MB.
    cache_dir : str or None
        Cache directory path.

    Examples
    --------
    >>> from samrfi.utils.model_cache import ModelCache
    >>>
    >>> # Initialize with default cache
    >>> cache = ModelCache()
    >>>
    >>> # Check if model is cached
    >>> is_cached = cache.is_cached('large')
    >>> print(f"Model cached: {is_cached}")
    >>>
    >>> # Get cache information
    >>> info = cache.get_cache_info('large')
    >>> print(f"Model ID: {info['model_id']}")
    >>> print(f"Size: {info['size_mb']} MB")
    >>> print(f"Cached: {info['is_cached']}")
    >>>
    >>> # Pre-download model with progress bar
    >>> cache.download_model('large', show_progress=True)
    Downloading SAM2 model 'large' (~850 MB)...
    Model ID: facebook/sam2-hiera-large
    Cache: ~/.cache/huggingface/hub/
    >>>
    >>> # Load model and processor (auto-downloads if needed)
    >>> model, processor = cache.load_model('large', device='cuda')
    >>>
    >>> # Use custom cache directory
    >>> custom_cache = ModelCache(cache_dir='/data/models')
    >>> model, processor = custom_cache.load_model('small')
    >>>
    >>> # List all available models
    >>> ModelCache.list_available_models()
    Available SAM2 models:
    ------------------------------------------------------------
      tiny         |   40 MB | facebook/sam2-hiera-tiny
      small        |  180 MB | facebook/sam2-hiera-small
      base_plus    |  330 MB | facebook/sam2-hiera-base-plus
      large        |  850 MB | facebook/sam2-hiera-large
    ------------------------------------------------------------
    Usage: ModelCache().load_model('checkpoint_name')
    """

    # Map checkpoint names to HuggingFace model IDs
    CHECKPOINT_MAP = {
        "tiny": "facebook/sam2-hiera-tiny",
        "small": "facebook/sam2-hiera-small",
        "base_plus": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }

    # Approximate model sizes (in MB)
    MODEL_SIZES = {
        "tiny": 40,
        "small": 180,
        "base_plus": 330,
        "large": 850,
    }

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        Initialize ModelCache.

        Parameters
        ----------
        cache_dir : str or None, optional
            Custom cache directory for model storage. If None, uses
            HuggingFace default (~/.cache/huggingface/hub/).
            Setting this also sets the HF_HOME environment variable.
            Default is None.

        Examples
        --------
        >>> # Use default cache
        >>> cache = ModelCache()
        >>>
        >>> # Use custom cache directory
        >>> cache = ModelCache(cache_dir='/data/models')
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir

    def get_model_id(self, checkpoint: str) -> str:
        """
        Get HuggingFace model ID for checkpoint name.

        Parameters
        ----------
        checkpoint : str
            Checkpoint name: 'tiny', 'small', 'base_plus', or 'large'.

        Returns
        -------
        str
            HuggingFace model ID (e.g., 'facebook/sam2-hiera-large').

        Raises
        ------
        ValueError
            If checkpoint name is not in CHECKPOINT_MAP.

        Examples
        --------
        >>> cache = ModelCache()
        >>> model_id = cache.get_model_id('large')
        >>> print(model_id)
        facebook/sam2-hiera-large
        >>>
        >>> # Invalid checkpoint raises error
        >>> cache.get_model_id('xlarge')
        Traceback (most recent call last):
            ...
        ValueError: Invalid checkpoint 'xlarge'. Valid options: tiny, small, base_plus, large
        """
        if checkpoint not in self.CHECKPOINT_MAP:
            valid = ", ".join(self.CHECKPOINT_MAP.keys())
            raise ValueError(f"Invalid checkpoint '{checkpoint}'. " f"Valid options: {valid}")
        return self.CHECKPOINT_MAP[checkpoint]

    def is_cached(self, checkpoint: str) -> bool:
        """
        Check if model is already cached locally.

        Checks for the existence of config.json in the local cache to
        determine if the model has been downloaded.

        Parameters
        ----------
        checkpoint : str
            Checkpoint name: 'tiny', 'small', 'base_plus', or 'large'.

        Returns
        -------
        bool
            True if model is cached locally, False otherwise.

        Examples
        --------
        >>> cache = ModelCache()
        >>> if cache.is_cached('large'):
        ...     print("Model already cached")
        ... else:
        ...     print("Model will be downloaded")
        """
        model_id = self.get_model_id(checkpoint)

        try:
            # Try to load from cache without downloading
            from transformers import cached_file

            # Check if config.json exists in cache
            config_path = cached_file(
                model_id,
                "config.json",
                cache_dir=self.cache_dir,
                local_files_only=True,
                _raise_exceptions_for_missing_entries=False,
            )
            return config_path is not None
        except Exception:
            return False

    def get_cache_info(self, checkpoint: str) -> Dict[str, any]:
        """
        Get cache information for a model.

        Returns comprehensive information about the model's cache status,
        including whether it's cached, its size, and cache path if available.

        Parameters
        ----------
        checkpoint : str
            Checkpoint name: 'tiny', 'small', 'base_plus', or 'large'.

        Returns
        -------
        dict
            Dictionary containing:
            - 'is_cached' : bool - Whether model is cached locally
            - 'model_id' : str - HuggingFace model ID
            - 'size_mb' : float - Approximate model size in MB
            - 'cache_path' : str - Local cache path (only if cached)

        Examples
        --------
        >>> cache = ModelCache()
        >>> info = cache.get_cache_info('large')
        >>> print(f"Model: {info['model_id']}")
        Model: facebook/sam2-hiera-large
        >>> print(f"Size: {info['size_mb']} MB")
        Size: 850 MB
        >>> print(f"Cached: {info['is_cached']}")
        Cached: True
        >>> if 'cache_path' in info:
        ...     print(f"Path: {info['cache_path']}")
        Path: /home/user/.cache/huggingface/hub/models--facebook--sam2-hiera-large
        """
        model_id = self.get_model_id(checkpoint)
        is_cached = self.is_cached(checkpoint)

        info = {
            "is_cached": is_cached,
            "model_id": model_id,
            "size_mb": self.MODEL_SIZES.get(checkpoint, 0),
        }

        if is_cached:
            # Try to find cache path
            try:
                from transformers import cached_file

                config_path = cached_file(
                    model_id, "config.json", cache_dir=self.cache_dir, local_files_only=True
                )
                if config_path:
                    info["cache_path"] = str(Path(config_path).parent)
            except Exception:
                pass

        return info

    def download_model(
        self, checkpoint: str, show_progress: bool = True, force_download: bool = False
    ) -> str:
        """
        Download model from HuggingFace Hub.

        Downloads the specified SAM2 model if not already cached. Shows
        progress bar by default and supports resume if interrupted.

        Parameters
        ----------
        checkpoint : str
            Checkpoint name: 'tiny', 'small', 'base_plus', or 'large'.
        show_progress : bool, optional
            If True, displays download progress bar and status messages.
            Default is True.
        force_download : bool, optional
            If True, re-downloads model even if already cached. Useful
            for updating to newer versions. Default is False.

        Returns
        -------
        str
            Path to the cached model directory.

        Notes
        -----
        - Downloads can be resumed if interrupted
        - Models are deduplicated using git-based storage in HuggingFace cache
        - First download may take several minutes depending on model size

        Examples
        --------
        >>> cache = ModelCache()
        >>>
        >>> # Download with progress (default)
        >>> path = cache.download_model('large')
        Downloading SAM2 model 'large' (~850 MB)...
        Model ID: facebook/sam2-hiera-large
        Cache: ~/.cache/huggingface/hub/
        [download progress bar]
        ✓ Download complete: /home/user/.cache/huggingface/hub/...
        >>>
        >>> # Silent download
        >>> path = cache.download_model('small', show_progress=False)
        >>>
        >>> # Force re-download
        >>> path = cache.download_model('large', force_download=True)
        """
        model_id = self.get_model_id(checkpoint)

        if not force_download and self.is_cached(checkpoint):
            info = self.get_cache_info(checkpoint)
            cache_path = info.get("cache_path", "unknown")
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
            tqdm_class=tqdm if show_progress else None,
        )

        if show_progress:
            print(f"✓ Download complete: {cache_path}")

        return cache_path

    def load_model(
        self, checkpoint: str, show_progress: bool = True, device: str = "cuda"
    ) -> Tuple[Sam2Model, Sam2Processor]:
        """
        Load SAM2 model and processor.

        Loads the specified SAM2 model and processor from cache, automatically
        downloading if not already cached. Moves model to specified device.

        Parameters
        ----------
        checkpoint : str
            Checkpoint name: 'tiny', 'small', 'base_plus', or 'large'.
        show_progress : bool, optional
            If True, shows progress messages during loading and download.
            Default is True.
        device : str, optional
            Device to load model on: 'cuda' for GPU or 'cpu' for CPU.
            Default is 'cuda'.

        Returns
        -------
        model : Sam2Model
            Loaded SAM2 model on specified device.
        processor : Sam2Processor
            SAM2 processor for input/output handling.

        Notes
        -----
        - First call downloads model (~40-850 MB depending on checkpoint)
        - Subsequent calls load from cache (much faster)
        - Model is automatically moved to specified device

        Examples
        --------
        >>> cache = ModelCache()
        >>>
        >>> # Load on GPU (default)
        >>> model, processor = cache.load_model('large')
        Model 'large' not found in cache.
        Downloading from HuggingFace (~850 MB)...
        This is a one-time download. Subsequent runs will use cached model.
        <BLANKLINE>
        Loading SAM2 processor...
        Loading SAM2 model...
        ✓ Model loaded: facebook/sam2-hiera-large
          Device: cuda
          Cache: /home/user/.cache/huggingface/hub/...
        >>>
        >>> # Load on CPU
        >>> model, processor = cache.load_model('small', device='cpu')
        >>>
        >>> # Silent loading
        >>> model, processor = cache.load_model('tiny', show_progress=False)
        """
        model_id = self.get_model_id(checkpoint)

        # Check cache status
        is_cached = self.is_cached(checkpoint)

        if not is_cached and show_progress:
            size_mb = self.MODEL_SIZES.get(checkpoint, 0)
            print(f"\nModel '{checkpoint}' not found in cache.")
            print(f"Downloading from HuggingFace (~{size_mb} MB)...")
            print("This is a one-time download. Subsequent runs will use cached model.\n")

        # Load model and processor (auto-downloads if needed)
        if show_progress and not is_cached:
            print("Loading SAM2 processor...")
        processor = Sam2Processor.from_pretrained(model_id, cache_dir=self.cache_dir)

        if show_progress and not is_cached:
            print("Loading SAM2 model...")
        model = Sam2Model.from_pretrained(model_id, cache_dir=self.cache_dir)

        # Move to device
        model = model.to(device)

        if show_progress:
            info = self.get_cache_info(checkpoint)
            cache_path = info.get("cache_path", "cache")
            print(f"✓ Model loaded: {model_id}")
            print(f"  Device: {device}")
            print(f"  Cache: {cache_path}\n")

        return model, processor

    def clear_cache(self, checkpoint: Optional[str] = None) -> None:
        """
        Clear model cache.

        Deletes cached model files to free disk space. Models will need to
        be re-downloaded when next requested.

        Parameters
        ----------
        checkpoint : str or None, optional
            Checkpoint to clear. If None, prints cache status for all
            models without deleting anything. Default is None.

        Warnings
        --------
        This permanently deletes cached model files. They will need to be
        re-downloaded from HuggingFace when next used.

        Examples
        --------
        >>> cache = ModelCache()
        >>>
        >>> # Show cache status (doesn't delete)
        >>> cache.clear_cache()
        Cache information:
          tiny         (  40 MB): ✗ not cached
          small        ( 180 MB): ✓ cached
          base_plus    ( 330 MB): ✗ not cached
          large        ( 850 MB): ✓ cached
        <BLANKLINE>
        To clear a specific model: clear_cache('checkpoint_name')
        >>>
        >>> # Clear specific model
        >>> cache.clear_cache('small')
        ✓ Cleared cache for 'small': /home/user/.cache/huggingface/hub/...
        >>>
        >>> # Verify deletion
        >>> cache.is_cached('small')
        False
        """
        if checkpoint is None:
            print("Cache information:")
            for ckpt in self.CHECKPOINT_MAP.keys():
                info = self.get_cache_info(ckpt)
                status = "✓ cached" if info["is_cached"] else "✗ not cached"
                print(f"  {ckpt:12} ({info['size_mb']:4.0f} MB): {status}")
            print("\nTo clear a specific model: clear_cache('checkpoint_name')")
            return

        info = self.get_cache_info(checkpoint)

        if not info["is_cached"]:
            print(f"Model '{checkpoint}' is not cached.")
            return

        cache_path = info.get("cache_path")
        if cache_path and os.path.exists(cache_path):
            import shutil

            shutil.rmtree(cache_path)
            print(f"✓ Cleared cache for '{checkpoint}': {cache_path}")
        else:
            print(f"Could not find cache path for '{checkpoint}'")

    @staticmethod
    def list_available_models() -> None:
        """
        Print list of available SAM2 models with sizes.

        Displays a formatted table showing all available SAM2 checkpoint names,
        their approximate sizes in MB, and HuggingFace model IDs.

        Notes
        -----
        This is a static method and can be called without instantiating ModelCache.

        Examples
        --------
        >>> from samrfi.utils.model_cache import ModelCache
        >>> ModelCache.list_available_models()
        Available SAM2 models:
        ------------------------------------------------------------
          tiny         |   40 MB | facebook/sam2-hiera-tiny
          small        |  180 MB | facebook/sam2-hiera-small
          base_plus    |  330 MB | facebook/sam2-hiera-base-plus
          large        |  850 MB | facebook/sam2-hiera-large
        ------------------------------------------------------------
        Usage: ModelCache().load_model('checkpoint_name')
        """
        print("Available SAM2 models:")
        print("-" * 60)
        for checkpoint, model_id in ModelCache.CHECKPOINT_MAP.items():
            size_mb = ModelCache.MODEL_SIZES.get(checkpoint, 0)
            print(f"  {checkpoint:12} | {size_mb:4.0f} MB | {model_id}")
        print("-" * 60)
        print("Usage: ModelCache().load_model('checkpoint_name')")
