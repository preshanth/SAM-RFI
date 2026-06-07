Quick Start
===========

This guide walks through a complete SAM-RFI workflow: generating synthetic training data, training a SAM2 model, and applying it to flag RFI.

1. Generate Synthetic Training Data
------------------------------------

Generate 1000 synthetic samples with physically realistic RFI::

    samrfi generate-data \
      --source synthetic \
      --config configs/synthetic_data.yaml \
      --output ./datasets/synthetic_p_band

This creates two datasets:

- ``exact_masks/`` - Perfect ground truth (train on this!)
- ``mad_masks/`` - MAD-based masks (for comparison)

Example config (``configs/synthetic_data.yaml``)::

    synthetic:
      num_samples: 1000
      num_channels: 1024
      num_times: 1024

      # Physical scales
      noise_mjy: 1.0              # 1 mJy noise
      rfi_power_min: 1000.0       # 1000 Jy RFI min
      rfi_power_max: 10000.0      # 10000 Jy RFI max

      # RFI types per sample
      rfi_type_counts:
        narrowband_persistent: 2
        broadband_persistent: 1
        frequency_sweep: 1
        narrowband_bursty: 2
        broadband_bursty: 1

      # Realism features
      enable_bandpass_rolloff: true
      polarization_correlation: 0.8

    processing:
      normalize_before_stretch: false
      normalize_after_stretch: false
      patch_size: 1024
      flag_sigma: 5

2. Train SAM2 Model
-------------------

Train on synthetic data with exact ground truth::

    samrfi train \
      --config configs/sam2_training.yaml \
      --dataset ./datasets/synthetic_p_band/exact_masks \
      --output ./models/sam2_rfi_v1

Training config (``configs/sam2_training.yaml``)::

    model:
      checkpoint: large           # tiny, small, base_plus, large
      freeze_encoders: true

    training:
      num_epochs: 10
      batch_size: 4
      learning_rate: 1.0e-5
      weight_decay: 0.0
      device: cuda

    output:
      dir_path: ./models/sam2_rfi_v1
      save_plots: true

Monitor training progress - loss curves are saved to ``models/sam2_rfi_v1/loss_plot.png``.

3. Generate Dataset from Real MS
---------------------------------

Generate training data from your measurement set::

    samrfi generate-data \
      --source ms \
      --config configs/ms_data.yaml \
      --output ./datasets/vla_pband

MS config (``configs/ms_data.yaml``)::

    ms:
      path: /path/to/observation.ms
      num_antennas: 5
      data_mode: DATA

    processing:
      patch_size: 1024
      flag_sigma: 5
      custom_flag: true         # Use existing MS flags

4. Apply Model to Flag RFI
---------------------------

Single-pass prediction (fast)::

    samrfi predict \
      --model ./models/sam2_rfi_v1/sam2_model_20250930_120000.pth \
      --input observation.ms

Iterative prediction (3 passes for deep cleaning)::

    samrfi predict \
      --model ./models/sam2_rfi_v1/sam2_model_20250930_120000.pth \
      --input observation.ms \
      --iterations 3

Iterative flagging progressively cleans deeper RFI:

- **Pass 1**: Finds bright RFI
- **Pass 2**: Masks bright RFI, finds hidden fainter RFI
- **Pass 3**: Final cleanup

Typically converges in 2-3 iterations.

Python API Usage
----------------

Load and Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from rfi_toolbox.io import MSLoader
    from rfi_toolbox.preprocessing import Preprocessor

    # Load measurement set
    loader = MSLoader('observation.ms')
    loader.load(num_antennas=5, mode='DATA')

    # Preprocess
    preprocessor = Preprocessor(loader.data, flags=loader.load_flags())
    dataset = preprocessor.create_dataset(
        patch_size=128,
        stretch='SQRT',
        flag_sigma=5
    )

    # Save for later
    dataset.save_to_disk('./my_dataset')

Train a Model
~~~~~~~~~~~~~

::

    from samrfi.training import SAM2Trainer
    from samrfi.data import NumpyDataset

    # Load dataset
    dataset = NumpyDataset.load('./my_dataset/exact_masks.npz')

    # Create trainer
    trainer = SAM2Trainer(dataset, device='cuda')

    # Train
    trainer.train(
        num_epochs=10,
        batch_size=4,
        sam_checkpoint='large',
        learning_rate=1e-5,
        plot=True
    )

Apply Trained Model
~~~~~~~~~~~~~~~~~~~

::

    from samrfi.inference import RFIPredictor

    # Load predictor
    predictor = RFIPredictor(
        model_path='./models/sam2_rfi.pth',
        sam_checkpoint='large',
        device='cuda'
    )

    # Single-pass prediction
    flags = predictor.predict_ms(
        ms_path='observation.ms',
        save_flags=True
    )

    # Iterative prediction (3 passes)
    flags = predictor.predict_iterative(
        ms_path='observation.ms',
        num_iterations=3,
        save_flags=True
    )

    print(f"Flagged {flags.sum()/flags.size*100:.2f}% of data")

Next Steps
----------

- See :doc:`api` for complete API reference
- Check ``scripts/QUICKSTART.md`` for experiment tracking workflow
- Read ``refactor_plan.md`` for architecture details
- Explore 4 experiment configs in ``configs/experiments/``

For advanced usage, training tips, and troubleshooting, see the main `README.md <https://github.com/preshanth/SAM-RFI/blob/main/README.md>`_.
