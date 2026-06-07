.. _api:

API Reference
=============

This page provides detailed API documentation for SAM-RFI v2.0.

.. contents:: Table of Contents
   :local:
   :depth: 2

Data Module
-----------

The ``samrfi.data`` module provides tools for loading, preprocessing, and managing radio astronomy data.

MSLoader
~~~~~~~~

.. autoclass:: rfi_toolbox.io.MSLoader
   :members:
   :undoc-members:
   :show-inheritance:

   Load CASA measurement sets and extract visibility data.

   **Example**::

       from rfi_toolbox.io import MSLoader

       loader = MSLoader('observation.ms')
       loader.load(num_antennas=5, mode='DATA')
       data = loader.data          # Complex visibilities
       magnitude = loader.magnitude  # Magnitude
       flags = loader.load_flags()

Preprocessor
~~~~~~~~~~~~

.. autoclass:: rfi_toolbox.preprocessing.Preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

   Preprocess visibility data for SAM2 training.

   **Example**::

       from rfi_toolbox.preprocessing import Preprocessor

       preprocessor = Preprocessor(data, flags=flags)
       dataset = preprocessor.create_dataset(
           patch_size=1024,
           flag_sigma=5,
           normalize_before_stretch=False
       )

SAMDataset
~~~~~~~~~~

.. autoclass:: samrfi.data.SAMDataset
   :members:
   :undoc-members:
   :show-inheritance:

   PyTorch Dataset wrapper for SAM2 training.

BatchedDataset
~~~~~~~~~~~~~~

.. autoclass:: samrfi.data.BatchedDataset
   :members:
   :undoc-members:
   :show-inheritance:

   Batched dataset for efficient large-scale training.

NumpyDataset
~~~~~~~~~~~~

.. autoclass:: samrfi.data.NumpyDataset
   :members:
   :undoc-members:
   :show-inheritance:

   Efficient numpy-backed dataset format (.npz files).

   **Example**::

       from samrfi.data import NumpyDataset

       # Load dataset
       dataset = NumpyDataset.load('dataset.npz')

       # Access data
       image = dataset[0]['image']
       label = dataset[0]['label']

BatchWriter
~~~~~~~~~~~

.. autoclass:: samrfi.data.BatchWriter
   :members:
   :undoc-members:
   :show-inheritance:

   Write datasets in batched format for memory efficiency.

HFDatasetWrapper
~~~~~~~~~~~~~~~~

.. autoclass:: samrfi.data.HFDatasetWrapper
   :members:
   :undoc-members:
   :show-inheritance:

   Convert between NumpyDataset and HuggingFace Dataset formats.

Data Generation Module
----------------------

The ``samrfi.data_generation`` module provides tools for generating training datasets.

SyntheticDataGenerator
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rfi_toolbox.data_generation.SyntheticDataGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   Generate physically realistic synthetic RFI data with exact ground truth.

   **Example**::

       from rfi_toolbox.data_generation import SyntheticDataGenerator

       # config provides the .synthetic / .processing sections (e.g. via ConfigLoader)
       generator = SyntheticDataGenerator(config)
       generator.generate(output_path='./datasets/synthetic')

   **RFI Types:**

   - Narrowband persistent (GPS, satellites)
   - Broadband persistent (power lines)
   - Narrowband bursty (pulsed transmitters)
   - Broadband bursty (lightning)
   - Frequency sweeps (radar chirps)

MSDataGenerator
~~~~~~~~~~~~~~~

.. autoclass:: samrfi.data_generation.MSDataGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   Convert CASA measurement sets to training datasets.

Training Module
---------------

The ``samrfi.training`` module provides SAM2 model training.

SAM2Trainer
~~~~~~~~~~~

.. autoclass:: samrfi.training.SAM2Trainer
   :members:
   :undoc-members:
   :show-inheritance:

   Train SAM2 models using HuggingFace transformers.

   **Example**::

       from samrfi.training import SAM2Trainer

       trainer = SAM2Trainer(dataset, device='cuda')
       trainer.train(
           num_epochs=10,
           batch_size=4,
           sam_checkpoint='large',
           learning_rate=1e-5,
           plot=True
       )

   **Features:**

   - Validation loss tracking
   - Best model checkpointing
   - Loss curve plotting
   - Configurable optimizer and loss functions

Inference Module
----------------

The ``samrfi.inference`` module provides trained model inference.

RFIPredictor
~~~~~~~~~~~~

.. autoclass:: samrfi.inference.RFIPredictor
   :members:
   :undoc-members:
   :show-inheritance:

   Apply trained SAM2 models to flag RFI in measurement sets.

   **Example**::

       from samrfi.inference import RFIPredictor

       predictor = RFIPredictor(
           model_path='model.pth',
           sam_checkpoint='large',
           device='cuda'
       )

       # Single-pass prediction
       flags = predictor.predict_ms('observation.ms')

       # Iterative prediction (N passes)
       flags = predictor.predict_iterative('observation.ms', num_iterations=3)

   **Iterative Flagging:**

   - Pass 1: Find bright RFI
   - Pass 2: Mask bright RFI, find hidden fainter RFI
   - Pass N: Progressive cleanup
   - Typically converges in 2-3 iterations

Config Module
-------------

The ``samrfi.config`` module provides configuration management.

ConfigLoader
~~~~~~~~~~~~

.. autoclass:: samrfi.config.ConfigLoader
   :members:
   :undoc-members:
   :show-inheritance:

   Load and validate YAML configuration files.

   **Example**::

       from samrfi.config import ConfigLoader

       # Load training config
       config = ConfigLoader.load_training('config.yaml')

       # Load data generation config
       config = ConfigLoader.load_data('data_config.yaml')

   **Config Types:**

   - ``TrainingConfig``: Training parameters (epochs, batch size, learning rate)
   - ``DataConfig``: Data generation parameters (nested structure)

Utilities Module
----------------

The ``samrfi.utils`` module provides utility functions and helpers.

ModelCache
~~~~~~~~~~

.. autoclass:: samrfi.utils.ModelCache
   :members:
   :undoc-members:
   :show-inheritance:

   Manage SAM2 model downloads and caching from HuggingFace.

   **Example**::

       from samrfi.utils import ModelCache

       cache = ModelCache()

       # Check if model is cached
       if cache.is_cached('large'):
           print("Model already downloaded")

       # Get cache info
       info = cache.get_cache_info('large')
       print(f"Size: {info['size_mb']} MB")

       # Pre-download model with progress bar
       cache.download_model('large', show_progress=True)

       # Load model (auto-downloads if needed)
       model, processor = cache.load_model('large')

   **Available Models:**

   - ``tiny`` - 40 MB
   - ``small`` - 180 MB
   - ``base_plus`` - 330 MB
   - ``large`` - 850 MB (recommended)

   Models are cached at: ``~/.cache/huggingface/hub/``

Command-Line Interface
----------------------

SAM-RFI provides a complete CLI via the ``samrfi`` command.

Generate Data
~~~~~~~~~~~~~

Generate synthetic or MS-based training datasets::

    samrfi generate-data --source {synthetic|ms} --config CONFIG.yaml --output DIR

Train Models
~~~~~~~~~~~~

Train SAM2 models with validation tracking::

    samrfi train --config CONFIG.yaml --dataset DIR [--validation-dataset VAL_DIR]

Predict RFI
~~~~~~~~~~~

Apply trained models to flag RFI::

    # Single-pass
    samrfi predict --model MODEL.pth --input OBS.ms

    # Iterative (N passes)
    samrfi predict --model MODEL.pth --input OBS.ms --iterations 3

Config Management
~~~~~~~~~~~~~~~~~

Create and validate configuration files::

    samrfi create-config --type {training|data} --output CONFIG.yaml
    samrfi validate-config --config CONFIG.yaml

For complete CLI documentation, run::

    samrfi --help
    samrfi COMMAND --help
