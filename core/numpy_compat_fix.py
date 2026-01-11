"""
Compatibility fix for loading models saved with different numpy versions.
This creates a shim to handle numpy._core references from numpy 2.x models.
"""

import sys
import numpy as np

# Check if numpy._core exists (numpy 2.x has it, numpy 1.x doesn't)
if not hasattr(np, "_core"):
    # Aggressive compatibility shim
    import types

    # Create _core module
    _core = types.ModuleType("numpy._core")
    sys.modules["numpy._core"] = _core
    np._core = _core

    # Map numpy._core.numeric DIRECTLY to numpy
    # This ensures that anything looking for numpy._core.numeric finds numpy
    sys.modules["numpy._core.numeric"] = np
    _core.numeric = np

    print("✓ Aggressively injected numpy._core.numeric -> numpy")

# Verify it works
if __name__ == "__main__":
    print("NumPy compatibility fix loaded")
    print(f"NumPy version: {np.__version__}")
    try:
        import numpy._core
        import numpy._core.numeric

        print("✓ numpy._core available")
        print("✓ numpy._core.numeric available")
    except Exception as e:
        print(f"✗ Error: {e}")
