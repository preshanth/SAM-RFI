"""
Model cache management for SAM-RFI

Handles downloading and caching SAM2 models from HuggingFace.
Provides progress bars and cache location management.
"""

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from tqdm import tqdm
    from transformers import Sam2Model, Sam2Processor
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Install with: pip install transformers huggingface_hub tqdm"
    ) from e

import fnmatch


class ModelCache:
    """
    Manage SAM2 model downloads and caching.

    SAM2 models are automatically downloaded from HuggingFace and cached locally.
    Default cache location: ~/.cache/huggingface/hub/

    Available models:
    - tiny: facebook/sam2-hiera-tiny (~40MB)
    - small: facebook/sam2-hiera-small (~180MB)
    - base_plus: facebook/sam2-hiera-base-plus (~330MB)
    - large: facebook/sam2-hiera-large (~850MB)

    Example:
        >>> from samrfi.utils import ModelCache
        >>>
        >>> # Check if model is cached
        >>> cache = ModelCache()
        >>> is_cached = cache.is_cached('large')
        >>>
        >>> # Get cache info
        >>> info = cache.get_cache_info('large')
        >>> print(f"Model size: {info['size_mb']:.1f} MB")
        >>>
        >>> # Pre-download model with progress bar
        >>> cache.download_model('large', show_progress=True)
        >>>
        >>> # Load model (auto-downloads if not cached)
        >>> model, processor = cache.load_model('large')
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

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize ModelCache.

        Args:
            cache_dir: Optional custom cache directory. If None, uses HuggingFace default
                      (~/.cache/huggingface/hub/)
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir

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
            valid = ", ".join(self.CHECKPOINT_MAP.keys())
            raise ValueError(f"Invalid checkpoint '{checkpoint}'. " f"Valid options: {valid}")
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
    ) -> tuple[Sam2Model, Sam2Processor]:
        """
        Load SAM2 model and processor (auto-downloads if not cached).

        Args:
            checkpoint: Checkpoint name (tiny, small, base_plus, large)
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

    def clear_cache(self, checkpoint: str | None = None) -> None:
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

    def list_repo_models(self, repo_id: str, pattern: str = "*.pth") -> list[dict]:
        """
        List model files in a HuggingFace repository.

        Args:
            repo_id: HuggingFace repo ID (e.g., 'polarimetric/sam-rfi')
            pattern: File pattern to filter (default: '*.pth')

        Returns:
            List of dicts with:
                - filename: str
                - size_mb: float (if available)

        Example:
            >>> cache = ModelCache()
            >>> models = cache.list_repo_models('polarimetric/sam-rfi')
            >>> for m in models:
            ...     print(f"{m['filename']:40} {m['size_mb']:8.1f} MB")
        """
        try:
            # List all files in repo
            all_files = list_repo_files(repo_id=repo_id, repo_type="model")

            # Filter by pattern
            matched_files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]

            # Build result list
            models = []
            for filename in matched_files:
                model_info = {"filename": filename, "size_mb": None}

                # Try to get file size (requires additional API call per file)
                # We'll skip this for now to keep it fast - size info not critical for listing
                models.append(model_info)

            return models

        except Exception as e:
            raise RuntimeError(
                f"Failed to list models in repository '{repo_id}': {e}\n"
                f"Make sure the repository exists and you have access to it.\n"
                f"For private repos, set HF_TOKEN environment variable."
            ) from e

    def download_from_repo(
        self,
        repo_id: str,
        filename: str,
        output_dir: str,
        local_name: str | None = None,
        show_progress: bool = True,
    ) -> str:
        """
        Download model from HuggingFace repo to custom directory.

        Args:
            repo_id: HuggingFace repo ID (e.g., 'polarimetric/sam-rfi')
            filename: Model filename in repo (e.g., 'sam2_rfi_v1.pth')
            output_dir: Local directory to save model
            local_name: Optional custom name (if None, uses original filename)
            show_progress: Show download progress bar

        Returns:
            Path to downloaded file

        Example:
            >>> cache = ModelCache()
            >>> # Download with original name
            >>> path = cache.download_from_repo(
            ...     'polarimetric/sam-rfi',
            ...     'sam2_v1.pth',
            ...     '/nfs/models/'
            ... )
            >>> # Download with custom name
            >>> path = cache.download_from_repo(
            ...     'polarimetric/sam-rfi',
            ...     'sam2_v1.pth',
            ...     '/nfs/models/',
            ...     local_name='my_custom_model.pth'
            ... )
        """
        # Create output directory if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine final filename
        final_name = local_name if local_name else filename
        final_path = output_path / final_name

        if show_progress:
            print("Downloading from HuggingFace...")
            print(f"  Repository: {repo_id}")
            print(f"  File: {filename}")
            print(f"  Destination: {final_path}")

        try:
            # Download file from HuggingFace
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                cache_dir=None,  # Use temp cache
                local_dir=None,
                local_dir_use_symlinks=False,
            )

            # Copy to final destination with desired name
            import shutil

            shutil.copy2(downloaded_path, final_path)

            if show_progress:
                size_mb = final_path.stat().st_size / (1024 * 1024)
                print(f"✓ Download complete ({size_mb:.1f} MB)")
                print(f"  Saved to: {final_path}")

            return str(final_path)

        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{filename}' from '{repo_id}': {e}\n"
                f"Make sure:\n"
                f"  1. Repository exists: https://huggingface.co/{repo_id}\n"
                f"  2. File exists in repository: {filename}\n"
                f"  3. You have access (set HF_TOKEN env var for private repos)\n"
                f"  4. Destination is writable: {output_dir}"
            ) from e

    @staticmethod
    def list_available_models() -> None:
        """Print list of available SAM2 models with sizes."""
        print("Available SAM2 models:")
        print("-" * 60)
        for checkpoint, model_id in ModelCache.CHECKPOINT_MAP.items():
            size_mb = ModelCache.MODEL_SIZES.get(checkpoint, 0)
            print(f"  {checkpoint:12} | {size_mb:4.0f} MB | {model_id}")
        print("-" * 60)
        print("Usage: ModelCache().load_model('checkpoint_name')")
