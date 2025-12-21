#!/usr/bin/env python
"""Test script to debug imports"""
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.path[:3]}")

try:
    print("Attempting to import torch...")
    import torch
    print(f"✓ torch imported: {torch.__version__}")
except ImportError as e:
    print(f"✗ Failed to import torch: {e}")
    sys.exit(1)

try:
    print("Attempting to import model.loss...")
    from model.loss import SigmoidContrastiveLoss
    print(f"✓ SigmoidContrastiveLoss imported")
except ImportError as e:
    print(f"✗ Failed to import loss: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Attempting to import model.siglip_model...")
    from model.siglip_model import SigLIPMedical
    print(f"✓ SigLIPMedical imported")
except ImportError as e:
    print(f"✗ Failed to import siglip_model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!")
