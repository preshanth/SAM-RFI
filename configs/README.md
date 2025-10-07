# Training Configurations

GPU-specific training configs optimized for different hardware setups.

## GPU Training Configs

| GPU | Config File | VRAM | System RAM | Batch Size | Model | Data Cache | Notes |
|-----|------------|------|------------|------------|-------|------------|-------|
| **GTX 1080 Ti** | `gpu_1080ti_training.yaml` | 11GB | 64GB | 4 | tiny | 40GB | Entry-level training |
| **V100** | `gpu_v100_training.yaml` | 32GB | 128GB | 12 | base_plus | 80GB | Good for prototyping |
| **A100** | `gpu_a100_training.yaml` | 80GB | 256GB | 24 | large | 180GB | Production training |
| **H100** | `h100_training_config.yaml` | 80GB | 230GB | 16 | large | 172GB | Fast training |
| **H200** | `gpu_h200_training.yaml` | 141GB | 512GB | 48 | large | 380GB | Maximum throughput |

## Dataset Generation Configs

| Config | Samples | Purpose |
|--------|---------|---------|
| `synthetic_test_100.yaml` | 100 | Quick testing |
| `synthetic_train_4k.yaml` | 4,000 | Development training |
| `synthetic_val_1k.yaml` | 1,000 | Validation |

## Key Parameters

### Data Caching Strategy

All GPU configs now use **shared RAM preloading** instead of per-worker LRU caching:

```yaml
training:
  data_cache_ram_gb: 172      # RAM budget for training data
  val_data_cache_ram_gb: 20   # RAM budget for validation data
```

**Benefits:**
- All workers share the same cached data (no duplication)
- Predictable memory usage
- Much faster than disk I/O once cached
- Configurable based on available system RAM

### Memory Calculations

Per batch file: **1.36 GB** (100 samples × 1024×1024×3 float32 + masks)

**Formula:**
```
batches_cached = data_cache_ram_gb / 1.36
percentage_cached = batches_cached / total_batches
```

**Example (H100):**
- Cache: 172 GB
- Batches: 172 / 1.36 = 126 batches
- For train_4000 (160 batches): 126/160 = **78% cached**
- For train_10000 (400 batches): 126/400 = **31% cached**

### DataLoader Settings

| Parameter | 1080 Ti | V100 | A100 | H100 | H200 |
|-----------|---------|------|------|------|------|
| `num_workers` | 4 | 8 | 16 | 12 | 24 |
| `prefetch_factor` | 2 | 2 | 3 | 2 | 4 |

## Usage

### Basic Training

```bash
# H100
python scripts/run_training.py --config configs/h100_training_config.yaml

# A100
python scripts/run_training.py --config configs/gpu_a100_training.yaml

# Skip dataset generation if already exists
python scripts/run_training.py --config configs/h100_training_config.yaml --skip-generation
```

### Monitor Training

Training logs are saved to:
```
{output_dir}/training_{timestamp}.log
```

Tail the log in another terminal:
```bash
tail -f ./training_output/training_*.log
```

### Adjusting for Your System

If you have different RAM than the config assumes:

1. **Check available RAM:**
   ```bash
   free -h
   ```

2. **Adjust `data_cache_ram_gb`:**
   - Use ~75% of available RAM for cache
   - Leave 25% for system/training overhead

3. **Example:** System with 128GB RAM:
   ```yaml
   data_cache_ram_gb: 96  # 75% of 128GB
   ```

## Model Checkpoints

| Model | Parameters | VRAM (approx) | Speed | Quality |
|-------|-----------|---------------|-------|---------|
| `tiny` | ~38M | ~5GB | Fastest | Good |
| `small` | ~80M | ~8GB | Fast | Better |
| `base_plus` | ~152M | ~15GB | Medium | Great |
| `large` | ~224M | ~22GB | Slower | Best |

## Common Issues

### OOM (Out of Memory)

**Symptom:** `DataLoader worker (pid XXX) is killed by signal: Killed`

**Solutions:**
1. Reduce `data_cache_ram_gb`
2. Reduce `num_workers`
3. Reduce `batch_size`

### Slow GPU Utilization

**Symptom:** GPU at 0% or low utilization

**Solutions:**
1. Increase `data_cache_ram_gb` (more data in RAM)
2. Increase `num_workers`
3. Increase `prefetch_factor`

### Cache Misses

**Symptom:** Frequent disk I/O, slow training

**Solutions:**
1. Increase `data_cache_ram_gb` to fit more of dataset
2. Use smaller dataset that fits in cache
3. Consider using faster storage (NVMe SSD)
