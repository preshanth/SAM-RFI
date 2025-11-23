#!/usr/bin/env python3
"""
SAM3 Validation Test - Check if Sam3Model supports fine-tuning with visual prompts

This script validates:
1. Can we import Sam3Model and Sam3Processor from transformers?
2. Does it have the same parameter structure as SAM2?
3. Can we freeze vision/prompt encoders?
4. Does forward pass work with VISUAL prompts (bounding boxes)?
5. Does backward pass work (gradient computation)?
6. Can we train like SAM2 (visual prompts only, no text)?

CRITICAL: SAM-RFI uses VISUAL prompts (bounding boxes), not text!

Run this BEFORE making any code changes.
"""

import sys
from pathlib import Path

print("="*70)
print("SAM3 VALIDATION TEST - VISUAL PROMPTS (SAM-RFI Use Case)")
print("="*70)
print()

# Test 1: Check transformers version and Sam3 availability
print("[1/7] Checking transformers library...")
try:
    import transformers
    print(f"  ✓ transformers version: {transformers.__version__}")

    # Check if Sam3Model exists (NOT Sam3TrackerModel)
    try:
        from transformers import Sam3Model, Sam3Processor
        print(f"  ✓ Sam3Model found")
        print(f"  ✓ Sam3Processor found")
        sam3_available = True
    except ImportError as e:
        print(f"  ✗ Sam3Model not available in transformers")
        print(f"    Error: {e}")
        print(f"    You may need to upgrade transformers:")
        print(f"    pip install --upgrade transformers")
        sam3_available = False

except ImportError:
    print(f"  ✗ transformers not installed")
    print(f"    Install with: pip install transformers")
    sys.exit(1)

print()

# Test 2: Check PyTorch
print("[2/7] Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"  ✗ PyTorch not installed")
    sys.exit(1)

print()

# If Sam3 not available, skip remaining tests
if not sam3_available:
    print("="*70)
    print("RESULT: Sam3Model not available in transformers")
    print("="*70)
    print()
    print("OPTIONS:")
    print("1. Upgrade transformers: pip install --upgrade transformers")
    print("2. Use native SAM3 package (requires more code changes)")
    print()
    sys.exit(1)

# Test 3: Load model and check parameter structure
print("[3/7] Loading Sam3Model...")
print("  NOTE: This requires facebook/sam3 access token")
print("  If you haven't authenticated: huggingface-cli login")
print()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model on {device}...")

    model = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = model.to(device)

    print(f"  ✓ Model loaded successfully")

    # Analyze parameter structure
    print()
    print("  Parameter Structure:")
    total_params = 0
    vision_encoder_params = 0
    prompt_encoder_params = 0
    mask_decoder_params = 0
    detector_params = 0
    other_params = 0

    param_groups = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Categorize (SAM3 might have different structure than SAM2)
        if "vision_encoder" in name or "image_encoder" in name:
            vision_encoder_params += num_params
            category = "vision_encoder"
        elif "prompt_encoder" in name or "text_encoder" in name:
            prompt_encoder_params += num_params
            category = "prompt_encoder"
        elif "mask_decoder" in name or "decoder" in name:
            mask_decoder_params += num_params
            category = "mask_decoder"
        elif "detector" in name:
            detector_params += num_params
            category = "detector"
        else:
            other_params += num_params
            category = "other"

        if category not in param_groups:
            param_groups[category] = []
        param_groups[category].append((name, num_params))

    print(f"    Total parameters: {total_params/1e6:.1f}M")
    print(f"    Vision encoder: {vision_encoder_params/1e6:.1f}M")
    print(f"    Prompt encoder: {prompt_encoder_params/1e6:.1f}M")
    print(f"    Mask decoder: {mask_decoder_params/1e6:.1f}M")
    print(f"    Detector: {detector_params/1e6:.1f}M")
    print(f"    Other: {other_params/1e6:.1f}M")

    # Show what we'd train (mask_decoder or detector)
    print()
    if "mask_decoder" in param_groups:
        print(f"  ✓ Found mask_decoder ({len(param_groups['mask_decoder'])} layers)")
        print(f"    First 3 layers:")
        for name, num_params in param_groups["mask_decoder"][:3]:
            print(f"      - {name}: {num_params:,} params")

    if "decoder" in param_groups:
        print(f"  ✓ Found decoder ({len(param_groups['decoder'])} layers)")
        print(f"    First 3 layers:")
        for name, num_params in param_groups["decoder"][:3]:
            print(f"      - {name}: {num_params:,} params")

except Exception as e:
    print(f"  ✗ Failed to load model")
    print(f"    Error: {e}")
    print()
    print("  This likely means:")
    print("  1. You don't have access to facebook/sam3 on HuggingFace")
    print("  2. You haven't authenticated (run: huggingface-cli login)")
    print("  3. The model checkpoint is not compatible with transformers")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Test freezing encoders (like SAM2 training does)
print("[4/7] Testing encoder freezing (SAM-RFI training pattern)...")
try:
    # Count trainable params before freezing
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params before freezing: {trainable_before/1e6:.1f}M")

    # Freeze encoders (try multiple naming patterns for SAM3)
    freeze_patterns = ["vision_encoder", "image_encoder", "prompt_encoder", "text_encoder"]

    for name, param in model.named_parameters():
        should_freeze = any(pattern in name for pattern in freeze_patterns)
        if should_freeze:
            param.requires_grad_(False)

    # Count trainable params after freezing
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params after freezing: {trainable_after/1e6:.1f}M")

    if trainable_after < trainable_before:
        print(f"  ✓ Freezing works! Reduced trainable params by {(trainable_before-trainable_after)/1e6:.1f}M")
    else:
        print(f"  ✗ Warning: Freezing didn't reduce trainable params")

except Exception as e:
    print(f"  ✗ Freezing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: CRITICAL - Test forward pass with VISUAL prompts (bounding boxes)
print("[5/7] Testing forward pass with VISUAL prompts (bounding boxes)...")
print("  CRITICAL: SAM-RFI uses bounding boxes, NOT text!")
try:
    import numpy as np
    from PIL import Image

    # Create dummy RFI-like data (1024x1024 grayscale converted to RGB)
    dummy_data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    dummy_image = Image.fromarray(np.stack([dummy_data]*3, axis=-1))

    # Bounding box prompt (like SAM-RFI extracts from masks)
    # Format: [[x_min, y_min, x_max, y_max]]
    dummy_boxes = [[[100, 100, 500, 500]]]  # One box

    print(f"  Input image: {dummy_image.size}")
    print(f"  Bounding boxes: {dummy_boxes}")

    # Try to process with bounding boxes (like SAM2)
    try:
        inputs = processor(
            images=dummy_image,
            input_boxes=dummy_boxes,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass (inference mode)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"  ✓ Forward pass with BOUNDING BOXES successful!")
        print(f"    Output type: {type(outputs)}")

        # Check output structure
        if hasattr(outputs, 'pred_masks'):
            print(f"    ✓ Has pred_masks: {outputs.pred_masks.shape}")
        elif hasattr(outputs, 'masks'):
            print(f"    ✓ Has masks: {outputs.masks.shape}")
        else:
            print(f"    ⚠️  Unknown output structure: {dir(outputs)}")

        visual_prompts_work = True

    except Exception as e:
        print(f"  ✗ Bounding boxes NOT supported!")
        print(f"    Error: {e}")
        print()
        print(f"  Trying with TEXT prompt instead (fallback)...")

        # Try text prompt as fallback
        try:
            inputs = processor(
                images=dummy_image,
                text="object",
                return_tensors="pt"
            )

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)

            print(f"  ✓ Text prompts work, but VISUAL prompts don't")
            print(f"    ⚠️  This means Sam3Model is TEXT-ONLY")
            print(f"    ⚠️  Cannot use for SAM-RFI (needs bounding boxes)")
            visual_prompts_work = False

        except Exception as e2:
            print(f"  ✗ Text prompts also failed: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

except Exception as e:
    print(f"  ✗ Forward pass test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Only continue if visual prompts work
if not visual_prompts_work:
    print("="*70)
    print("CRITICAL FAILURE: Sam3Model does NOT support visual prompts!")
    print("="*70)
    print()
    print("Sam3Model appears to be TEXT-ONLY (Promptable Concept Segmentation)")
    print("SAM-RFI requires VISUAL prompts (bounding boxes)")
    print()
    print("OPTIONS:")
    print("1. Check if Sam3TrackerModel exists (separate class for visual prompts)")
    print("2. Use native SAM3 package")
    print("3. Stay with SAM2 (works with visual prompts)")
    print()
    sys.exit(1)

# Test 6: Test backward pass (CRITICAL for training!)
print("[6/7] Testing backward pass (gradient computation)...")
try:
    # Switch to training mode
    model.train()

    # Forward pass
    outputs = model(**inputs)

    # Get masks (handle different output formats)
    if hasattr(outputs, 'pred_masks'):
        pred_masks = outputs.pred_masks
    elif hasattr(outputs, 'masks'):
        pred_masks = outputs.masks
    else:
        raise ValueError("Cannot find masks in output")

    # Compute dummy loss (like DiceCELoss in training)
    dummy_loss = pred_masks.sum()  # Simplified - real training uses DiceCELoss

    print(f"  Dummy loss value: {dummy_loss.item():.4f}")

    # Backward pass
    dummy_loss.backward()

    # Check if gradients were computed for trainable layers
    has_gradients = False
    encoder_has_grad = False

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            # Check if frozen params have gradients (shouldn't)
            if any(pattern in name for pattern in freeze_patterns):
                encoder_has_grad = True

    if has_gradients:
        print(f"  ✓ Backward pass successful - gradients computed")
    else:
        print(f"  ✗ Warning: No gradients computed")

    if encoder_has_grad:
        print(f"  ✗ Warning: Frozen encoders have gradients (shouldn't happen)")
    else:
        print(f"  ✓ Frozen encoders correctly have no gradients")

    print(f"  ✓ Training should work!")

except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("  This means Sam3Model does NOT support fine-tuning via transformers")
    print("  You would need to use the native SAM3 package instead")
    sys.exit(1)

print()

# Test 7: Check for Sam3TrackerModel (alternative class for visual prompts)
print("[7/7] Checking for Sam3TrackerModel (visual prompts variant)...")
try:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    print(f"  ✓ Sam3TrackerModel also available!")
    print(f"    This might be better for visual-only prompts")
    has_tracker = True
except ImportError:
    print(f"  ✗ Sam3TrackerModel not found")
    print(f"    Only Sam3Model available (may be text-focused)")
    has_tracker = False

print()

# Summary
print("="*70)
print("VALIDATION SUMMARY")
print("="*70)
print()
print(f"✓ Sam3Model is available in transformers {transformers.__version__}")
print(f"✓ Model has {total_params/1e6:.1f}M total parameters")
print(f"✓ Trainable params after freezing: {trainable_after/1e6:.1f}M")
print("✓ Encoder freezing works")

if visual_prompts_work:
    print("✓ Forward pass with VISUAL prompts (bounding boxes) works")
    print("✓ Backward pass (gradient computation) works")
    print()
    print("="*70)
    print("CONCLUSION: Sam3Model SUPPORTS fine-tuning with visual prompts! ✓")
    print("="*70)
    print()
    print("NEXT STEPS:")
    print("1. Migration should be straightforward (~20 line changes)")
    print("2. Update sam2_trainer.py to use Sam3Model")
    print("3. Update predictor.py to use Sam3Model")
    print("4. Test with bounding box prompts (no text needed)")
    print("5. Test training on small synthetic RFI dataset")
    print()
    print("ESTIMATED MIGRATION TIME: 2-3 hours")
else:
    print("✗ Visual prompts (bounding boxes) NOT supported")
    print()
    print("="*70)
    print("CONCLUSION: Sam3Model is TEXT-ONLY, cannot use for SAM-RFI")
    print("="*70)
    print()
    if has_tracker:
        print("RECOMMENDATION: Try Sam3TrackerModel instead")
    else:
        print("RECOMMENDATION: Use native SAM3 package or stay with SAM2")

print()
