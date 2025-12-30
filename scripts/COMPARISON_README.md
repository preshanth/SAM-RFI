# SAM-RFI Flagging Comparison

Compare SAM-RFI vs CASA (tfcrop+rflag) with image quality metrics.

## Quick Start

```bash
# 1. Download tutorial data (26 GB)
python compare_flagging_methods.py --download-data --data-dir ./data

# 2. Prepare MS in CASA
casa -c prepare_tutorial_data.py --download --data-dir ./data --output ./processed

# 3. Check compatibility (optional)
python check_ms_compatibility.py processed/3C129_pband.ms --patch-size 1024

# 4. Run comparison in CASA
casa -c compare_flagging_methods.py \
    --ms processed/3C129_pband.ms \
    --model /path/to/model.pth \
    --output ./results/
```

## Output Files

```
results/
├── image_casa.image.tt0          # CASA-flagged image
├── image_sam.image.tt0           # SAM-flagged image
├── plot_flag_percentage.png      # Bar chart
├── plot_flag_overlap.png         # Venn diagram
├── plot_image_metrics.png        # RMS + dynamic range
└── comparison_report.txt         # All metrics
```

## Key Metrics

**Visibility-level:**
- Flag % (CASA vs SAM)
- Overlap analysis
- Agreement %

**Image-level:**
- RMS noise (lower is better)
- Dynamic range (higher is better)
- % Improvement

## What Was Fixed

1. ✓ Added gencal calls (antpos, rq, tecim)
2. ✓ Use all SPWs (0~15) not just 3~8
3. ✓ Adaptive patching for arbitrary MS dimensions
4. ✓ Non-interactive tclean

## Troubleshooting

**TEC download fails?** Need internet, or it's skipped automatically.

**MS not divisible by 1024?** Adaptive padding handles it automatically.

**Out of memory?** Reduce batch_size=8 to 4 in comparison script line ~413.

## Scripts

- `compare_flagging_methods.py` - Main comparison pipeline
- `prepare_tutorial_data.py` - Automate CASA preprocessing
- `check_ms_compatibility.py` - Check MS dimensions
- `flag_3c391.py` - Simple inference example

Done. Ready to test.
