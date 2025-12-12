# Deprecated Files

These files have been deprecated in favor of the unified CLI workflow.

## Why deprecated:
- train_sam2.py: References non-existent NumpyDataset class, uses .npz format
- run_training.py: Duplicates CLI functionality with added complexity  
- experiments/: All configs reference deprecated .npz format

## Migration:
Use the CLI instead:
```bash
samrfi generate-data --source synthetic --config configs/synthetic_train_4k.yaml --output ./datasets/train
samrfi train --config configs/gpu_v100_training.yaml --dataset ./datasets/train/exact_masks --validation-dataset ./datasets/val/exact_masks
```

