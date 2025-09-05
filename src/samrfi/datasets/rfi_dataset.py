"""
HuggingFace-Compatible RFI Dataset Creator

Creates standardized datasets for RFI detection training that can be easily
shared via HuggingFace Hub and used across different training pipelines.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime

try:
    from datasets import Dataset, DatasetDict, Features, Image, Value, Array2D
    from huggingface_hub import HfApi
    from PIL import Image as PILImage

    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    Dataset = None
    DatasetDict = None
    logging.warning(f"HuggingFace dependencies not available: {e}")

logger = logging.getLogger(__name__)


class RFIDatasetCreator:
    """Create HuggingFace-compatible RFI detection datasets"""

    def __init__(self, output_dir: str = "rfi_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace dependencies not available. Please install datasets and huggingface-hub."
            )

    def create_training_dataset(
        self,
        ms_paths: List[str] = None,
        synthetic_samples: int = 500,
        train_split: float = 0.8,
        image_size: int = 1024,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Create standardized training dataset

        Args:
            ms_paths: List of measurement set paths (if available)
            synthetic_samples: Number of synthetic samples to generate
            train_split: Fraction of data for training
            image_size: Target image size for SAM
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with train/validation splits
        """
        np.random.seed(seed)

        # For now, create synthetic data (MS integration in Phase 2)
        logger.info(f"Creating {synthetic_samples} synthetic RFI samples")

        # Generate synthetic data
        synthetic_data = self._generate_synthetic_data(synthetic_samples, image_size)

        # Add real MS data if available (placeholder for Phase 2)
        if ms_paths:
            logger.info(f"MS data integration will be implemented in Phase 2")
            # real_data = self._extract_real_data(ms_paths, image_size)
            # all_data = synthetic_data + real_data
        else:
            all_data = synthetic_data

        # Create train/val split
        split_idx = int(len(all_data) * train_split)
        np.random.shuffle(all_data)

        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        logger.info(
            f"Created dataset: {len(train_data)} train, {len(val_data)} validation samples"
        )

        # Convert to HuggingFace format
        train_dataset = self._create_hf_dataset(train_data)
        val_dataset = self._create_hf_dataset(val_data)

        dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

        # Save metadata
        self._save_dataset_metadata(
            dataset_dict,
            {
                "synthetic_samples": synthetic_samples,
                "train_split": train_split,
                "image_size": image_size,
                "seed": seed,
                "creation_date": datetime.now().isoformat(),
            },
        )

        return dataset_dict

    def _generate_synthetic_data(
        self, num_samples: int, image_size: int
    ) -> List[Dict[str, Any]]:
        """Generate synthetic RFI data"""
        data = []

        for i in range(num_samples):
            # Create synthetic waterfall plot
            image, mask, metadata = self._create_synthetic_sample(image_size)

            # Convert to PIL Images for HF compatibility
            image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))

            sample = {
                "image": image_pil,
                "mask": mask_pil,
                "metadata": {
                    "source": "synthetic",
                    "sample_id": i,
                    "rfi_type": metadata["rfi_type"],
                    "rfi_intensity": float(metadata["rfi_intensity"]),
                    "frequency_range": metadata["frequency_range"],
                    "time_range": metadata["time_range"],
                    "image_size": image_size,
                },
            }

            data.append(sample)

        return data

    def _create_synthetic_sample(
        self, image_size: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create a single synthetic RFI sample"""
        # Create base noise pattern
        image = np.random.normal(0.1, 0.05, (image_size, image_size))
        image = np.clip(image, 0, 1)

        # Initialize mask
        mask = np.zeros((image_size, image_size))

        # Add different types of RFI
        rfi_types = ["broadband", "narrowband", "periodic", "intermittent"]
        rfi_type = np.random.choice(rfi_types)

        if rfi_type == "broadband":
            # Broadband RFI across frequency channels
            start_freq = np.random.randint(0, image_size // 2)
            end_freq = start_freq + np.random.randint(image_size // 4, image_size // 2)
            start_time = np.random.randint(0, image_size // 4)
            end_time = start_time + np.random.randint(image_size // 8, image_size // 4)

            intensity = np.random.uniform(0.3, 0.8)
            image[start_time:end_time, start_freq:end_freq] += intensity
            mask[start_time:end_time, start_freq:end_freq] = 1

        elif rfi_type == "narrowband":
            # Narrowband RFI - single frequency
            freq_channel = np.random.randint(0, image_size)
            start_time = np.random.randint(0, image_size // 2)
            end_time = start_time + np.random.randint(image_size // 4, image_size // 2)

            intensity = np.random.uniform(0.4, 0.9)
            width = np.random.randint(1, 5)

            for w in range(width):
                if freq_channel + w < image_size:
                    image[start_time:end_time, freq_channel + w] += intensity
                    mask[start_time:end_time, freq_channel + w] = 1

        elif rfi_type == "periodic":
            # Periodic RFI
            period = np.random.randint(10, 50)
            num_channels = np.random.randint(5, 20)

            for t in range(0, image_size, period):
                end_t = min(t + period // 4, image_size)
                start_f = np.random.randint(0, image_size - num_channels)
                end_f = start_f + num_channels

                intensity = np.random.uniform(0.2, 0.6)
                image[t:end_t, start_f:end_f] += intensity
                mask[t:end_t, start_f:end_f] = 1

        elif rfi_type == "intermittent":
            # Intermittent bursts
            num_bursts = np.random.randint(3, 8)

            for _ in range(num_bursts):
                t_center = np.random.randint(0, image_size)
                f_center = np.random.randint(0, image_size)
                t_width = np.random.randint(5, 20)
                f_width = np.random.randint(5, 20)

                t_start = max(0, t_center - t_width // 2)
                t_end = min(image_size, t_center + t_width // 2)
                f_start = max(0, f_center - f_width // 2)
                f_end = min(image_size, f_center + f_width // 2)

                intensity = np.random.uniform(0.3, 0.7)
                image[t_start:t_end, f_start:f_end] += intensity
                mask[t_start:t_end, f_start:f_end] = 1

        # Clip values
        image = np.clip(image, 0, 1)

        # Metadata
        metadata = {
            "rfi_type": rfi_type,
            "rfi_intensity": float(np.mean(image[mask == 1]) if np.any(mask) else 0.0),
            "frequency_range": [0.0, 1.0],  # Normalized
            "time_range": [0.0, 1.0],  # Normalized
            "rfi_fraction": float(np.sum(mask) / mask.size),
        }

        return image, mask, metadata

    def _create_hf_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """Convert data to HuggingFace Dataset format"""
        features = Features(
            {
                "image": Image(),
                "mask": Image(),
                "metadata": {
                    "source": Value("string"),
                    "sample_id": Value("int32"),
                    "rfi_type": Value("string"),
                    "rfi_intensity": Value("float32"),
                    "frequency_range": [Value("float32")],
                    "time_range": [Value("float32")],
                    "image_size": Value("int32"),
                },
            }
        )

        return Dataset.from_list(data, features=features)

    def _save_dataset_metadata(
        self, dataset: DatasetDict, metadata: Dict[str, Any]
    ) -> None:
        """Save dataset metadata to file"""
        metadata_path = self.output_dir / "dataset_metadata.json"

        # Add dataset statistics
        train_stats = self._compute_dataset_stats(dataset["train"])
        val_stats = self._compute_dataset_stats(dataset["validation"])

        full_metadata = {
            **metadata,
            "train_stats": train_stats,
            "validation_stats": val_stats,
            "total_samples": len(dataset["train"]) + len(dataset["validation"]),
        }

        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info(f"Saved dataset metadata to {metadata_path}")

    def _compute_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """Compute statistics for a dataset"""
        rfi_types = [sample["metadata"]["rfi_type"] for sample in dataset]
        rfi_intensities = [sample["metadata"]["rfi_intensity"] for sample in dataset]

        return {
            "num_samples": len(dataset),
            "rfi_type_distribution": {
                rfi_type: rfi_types.count(rfi_type) for rfi_type in set(rfi_types)
            },
            "mean_rfi_intensity": float(np.mean(rfi_intensities)),
            "std_rfi_intensity": float(np.std(rfi_intensities)),
        }

    def push_to_hub(
        self,
        dataset: DatasetDict,
        repo_name: str,
        private: bool = False,
        commit_message: str = None,
    ) -> str:
        """
        Push dataset to HuggingFace Hub

        Args:
            dataset: Dataset to upload
            repo_name: Repository name on HF Hub
            private: Whether to make repository private
            commit_message: Custom commit message

        Returns:
            URL to the uploaded dataset
        """
        if commit_message is None:
            commit_message = (
                f"Add SAM-RFI training dataset ({datetime.now().strftime('%Y-%m-%d')})"
            )

        try:
            dataset.push_to_hub(
                repo_name, private=private, commit_message=commit_message
            )

            url = f"https://huggingface.co/datasets/{repo_name}"
            logger.info(f"Dataset uploaded successfully: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload dataset to HF Hub: {e}")
            raise

    def load_from_hub(self, repo_name: str) -> DatasetDict:
        """Load dataset from HuggingFace Hub"""
        try:
            from datasets import load_dataset

            dataset = load_dataset(repo_name)
            logger.info(f"Loaded dataset from HF Hub: {repo_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from HF Hub: {e}")
            raise

    def create_model_card(self, dataset: DatasetDict, metadata: Dict[str, Any]) -> str:
        """Create a model card for the dataset"""
        train_stats = self._compute_dataset_stats(dataset["train"])
        val_stats = self._compute_dataset_stats(dataset["validation"])

        card_content = f"""
# SAM-RFI Training Dataset

## Dataset Description

This dataset contains synthetic Radio Frequency Interference (RFI) patterns for training SAM-based RFI detection models.

## Dataset Statistics

- **Total Samples**: {len(dataset['train']) + len(dataset['validation'])}
- **Training Samples**: {len(dataset['train'])}
- **Validation Samples**: {len(dataset['validation'])}
- **Image Size**: {metadata.get('image_size', 1024)}x{metadata.get('image_size', 1024)}

## RFI Types

The dataset includes the following types of synthetic RFI:

{self._format_rfi_distribution(train_stats['rfi_type_distribution'])}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/sam-rfi-dataset")

# Access training data
train_data = dataset['train']
print(f"Number of training samples: {{len(train_data)}}")

# Access a sample
sample = train_data[0]
image = sample['image']  # PIL Image
mask = sample['mask']    # PIL Image  
metadata = sample['metadata']  # Dictionary with RFI information
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{sam_rfi_dataset,
  title={{SAM-RFI: Synthetic RFI Dataset for Radio Astronomy}},
  author={{Deal, Derod and Jagannathan, Preshanth}},
  year={{2025}},
  url={{https://github.com/preshanth/SAM-RFI}}
}}
```

## License

MIT License
"""

        return card_content.strip()

    def _format_rfi_distribution(self, distribution: Dict[str, int]) -> str:
        """Format RFI type distribution for model card"""
        lines = []
        for rfi_type, count in distribution.items():
            lines.append(f"- **{rfi_type.capitalize()}**: {count} samples")
        return "\n".join(lines)
