Installation
============

Prerequisites
-------------

SAM-RFI requires:

- Python 3.10, 3.11, or 3.12
- CUDA-capable GPU (recommended for training)
- Git (for cloning repository)

Quick Install
-------------

1. Clone the repository::

    git clone https://github.com/preshanth/SAM-RFI.git
    cd SAM-RFI

2. Create a conda environment::

    conda create -n samrfi python=3.12 -y
    conda activate samrfi

3. Fix pandas/numpy compatibility (important!)::

    pip install pandas>=2.2.0 numpy>=1.26.0 --only-binary :all:

4. Install SAM-RFI with all dependencies::

    pip install -e .[dev]

This installs:

- **Core dependencies**: numpy, scipy, pandas, matplotlib, PyYAML
- **Deep learning**: torch, transformers, monai
- **Data handling**: HuggingFace datasets
- **CASA tools**: casatools, casatasks
- **Development**: pytest, black, mypy, flake8
- **GPU profiling**: nvidia-ml-py3 (pynvml)

Verify Installation
-------------------

Test that the CLI is available::

    samrfi --help

Test imports::

    python -c "from samrfi.data import MSLoader, Preprocessor; from samrfi.training import SAM2Trainer; print('✓ Installation successful')"

Installation Options
--------------------

Minimal Install (inference only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only need to apply pre-trained models::

    pip install -e .

This installs core dependencies without development tools.

GPU Support
~~~~~~~~~~~

SAM-RFI automatically detects CUDA GPUs. For CPU-only systems, training will be slower but still functional.

To verify GPU access::

    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Model Auto-Download
~~~~~~~~~~~~~~~~~~~

**SAM2 models are automatically downloaded from HuggingFace on first use.**

When you first train or run inference, SAM-RFI will:

1. Check if the model is cached locally (``~/.cache/huggingface/hub/``)
2. If not found, download from HuggingFace (one-time, with progress bar)
3. Cache for future use

**Available models:**

- ``tiny`` - 40 MB (fastest, lower accuracy)
- ``small`` - 180 MB (balanced)
- ``base_plus`` - 330 MB (good accuracy)
- ``large`` - 850 MB (best accuracy, recommended)

**Pre-download models** (optional)::

    python -c "from samrfi.utils import ModelCache; ModelCache().download_model('large')"

**Custom cache location** (optional)::

    export HF_HOME=/path/to/custom/cache
    samrfi train --config config.yaml --dataset dataset.npz

Common Issues
-------------

**ImportError: No module named 'casatools'**

CASA tools are installed automatically. If this error persists, try::

    pip install --upgrade casatools casatasks

**Version conflicts with numpy/pandas**

Run the pandas/numpy fix from step 3 above::

    pip install pandas>=2.2.0 numpy>=1.26.0 --only-binary :all:

**CUDA out of memory during training**

Reduce batch size in your training config::

    training:
      batch_size: 2  # or even 1 for small GPUs

See the :doc:`quickstart` guide for your first training run.
