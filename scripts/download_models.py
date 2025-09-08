#!/usr/bin/env python3
"""
Download SAM2 models to local models/ directory
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import Sam2Model, Sam2Processor
    from huggingface_hub import snapshot_download
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAM2 model variants and their HuggingFace model IDs
SAM2_MODELS = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small", 
    "base_plus": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large"
}

def download_sam2_model(variant: str, models_dir: Path, force_redownload: bool = False):
    """Download a specific SAM2 model variant"""
    if variant not in SAM2_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(SAM2_MODELS.keys())}")
    
    model_id = SAM2_MODELS[variant]
    local_dir = models_dir / f"sam2-{variant}"
    
    # Check if already exists
    if local_dir.exists() and not force_redownload:
        logger.info(f"Model {variant} already exists at {local_dir}")
        return local_dir
    
    logger.info(f"Downloading SAM2 {variant} from {model_id}")
    logger.info(f"Saving to: {local_dir}")
    
    try:
        # Download using snapshot_download for full model
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False  # Use actual files, not symlinks
        )
        
        # Verify the download by attempting to load
        logger.info(f"Verifying download for {variant}...")
        model = Sam2Model.from_pretrained(str(local_dir))
        processor = Sam2Processor.from_pretrained(str(local_dir))
        
        logger.info(f"✅ Successfully downloaded and verified SAM2 {variant}")
        logger.info(f"   Model path: {local_dir}")
        logger.info(f"   Model parameters: ~{get_model_size_estimate(variant)}")
        
        return local_dir
        
    except Exception as e:
        logger.error(f"❌ Failed to download {variant}: {e}")
        # Clean up partial download
        if local_dir.exists():
            import shutil
            shutil.rmtree(local_dir)
        raise

def get_model_size_estimate(variant: str) -> str:
    """Get estimated model size"""
    sizes = {
        "tiny": "~38MB",
        "small": "~184MB", 
        "base_plus": "~615MB",
        "large": "~2.4GB"
    }
    return sizes.get(variant, "Unknown")

def update_v100_config(models_dir: Path, variant: str):
    """Update V100 config to point to local model"""
    config_path = project_root / "configs" / "training" / "v100_config.yaml"
    local_model_path = models_dir / f"sam2-{variant}"
    
    if not config_path.exists():
        logger.warning(f"V100 config not found: {config_path}")
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update local_model_path
    import re
    pattern = r'local_model_path:\s*"[^"]*"'
    replacement = f'local_model_path: "{local_model_path}"'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        with open(config_path, 'w') as f:
            f.write(new_content)
        logger.info(f"Updated V100 config to use local model: {local_model_path}")
    else:
        logger.warning("Could not find local_model_path in V100 config")

def main():
    parser = argparse.ArgumentParser(description="Download SAM2 models to local directory")
    parser.add_argument(
        "--variant", 
        choices=list(SAM2_MODELS.keys()) + ["all"],
        default="large",
        help="SAM2 variant to download (default: large)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_root / "models",
        help="Directory to save models (default: ./models/)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--update-config",
        action="store_true", 
        help="Update V100 config to point to downloaded model"
    )
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers and huggingface_hub required. Install with:")
        logger.error("pip install transformers huggingface_hub")
        return 1
    
    # Create models directory
    args.models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {args.models_dir}")
    
    # Download models
    downloaded_models = []
    variants_to_download = list(SAM2_MODELS.keys()) if args.variant == "all" else [args.variant]
    
    for variant in variants_to_download:
        try:
            model_path = download_sam2_model(variant, args.models_dir, args.force)
            downloaded_models.append((variant, model_path))
        except Exception as e:
            logger.error(f"Failed to download {variant}: {e}")
            if args.variant != "all":  # Exit on single model failure
                return 1
    
    # Update config if requested
    if args.update_config and downloaded_models:
        # Use the requested variant, or large if downloading all
        config_variant = args.variant if args.variant != "all" else "large"
        if config_variant in [v for v, _ in downloaded_models]:
            update_v100_config(args.models_dir, config_variant)
    
    # Summary
    logger.info(f"\n🎉 Download complete!")
    logger.info(f"Downloaded {len(downloaded_models)} model(s):")
    for variant, path in downloaded_models:
        logger.info(f"  - {variant}: {path}")
    
    if downloaded_models:
        logger.info(f"\nTo use local models, set in your config:")
        logger.info(f"  local_model_path: \"{args.models_dir}/sam2-{{variant}}\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())