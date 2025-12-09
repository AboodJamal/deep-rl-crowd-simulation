"""
Compatibility fix for loading models saved with different numpy versions.
This creates a shim to handle numpy._core references from numpy 2.x models.
"""
import sys
import numpy as np

# Check if numpy._core exists (numpy 2.x has it, numpy 1.x doesn't)
if not hasattr(np, '_core'):
    # Create compatibility shim for numpy 1.x to load models saved with numpy 2.x
    import types
    
    # Create _core module structure
    _core_module = types.ModuleType('_core')
    
    # Create numeric submodule
    numeric_module = types.ModuleType('numeric')
    
    # Copy all numpy functions/attributes to numeric (including private ones)
    for attr in dir(np):
        try:
            # Include both public and private attributes (like _frombuffer)
            setattr(numeric_module, attr, getattr(np, attr))
        except:
            pass
    
    # Specifically add _frombuffer if it exists
    if hasattr(np, 'frombuffer'):
        numeric_module._frombuffer = np.frombuffer
    elif hasattr(np.core.multiarray, '_frombuffer'):
        numeric_module._frombuffer = np.core.multiarray._frombuffer
    
    # Make numeric available as _core.numeric
    _core_module.numeric = numeric_module
    
    # Also make it directly accessible
    _core_module.numeric = np
    
    # Attach to numpy
    np._core = _core_module
    
    # Also add to sys.modules for import resolution
    sys.modules['numpy._core'] = _core_module
    sys.modules['numpy._core.numeric'] = numeric_module

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

