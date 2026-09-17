#!/usr/bin/env python3
"""Test Taichi GPU configuration"""

import sys
import os

# Add taichi replication to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/taichi/replication'))

def test_default_arch():
    """Test that default architecture is GPU"""
    import runtime as rt
    assert rt.ARCH == "gpu", f"Default ARCH should be 'gpu', got '{rt.ARCH}'"
    print("✓ Default architecture is GPU")

def test_gpu_memory_fraction():
    """Test that GPU memory fraction is configurable"""
    import runtime as rt
    assert rt.GPU_MEMORY_FRACTION == 0.4, f"Default GPU_MEMORY_FRACTION should be 0.4, got {rt.GPU_MEMORY_FRACTION}"
    print(f"✓ GPU memory fraction set to {rt.GPU_MEMORY_FRACTION}")

def test_taichi_initialization():
    """Test that Taichi initializes with GPU backend"""
    import runtime as rt
    import simulation as sim

    # Check the backend that was selected
    backend = sim._GPU_BACKEND
    assert backend in ("cuda", "vulkan"), f"Backend should be cuda or vulkan, got {backend}"
    print(f"✓ Taichi initialized with {backend.upper()} backend")

    # Verify the arch setting
    assert rt.ARCH == "gpu", f"Runtime ARCH should be 'gpu', got '{rt.ARCH}'"
    print("✓ Runtime configured for GPU")

    return backend

def test_taichi_fields():
    """Test that basic Taichi fields can be created and used"""
    import taichi as ti
    import numpy as np

    # Create a simple field
    x = ti.field(ti.f32, shape=100)

    @ti.kernel
    def fill_field():
        for i in x:
            x[i] = float(i)

    fill_field()
    result = x.to_numpy()
    expected = np.arange(100, dtype=np.float32)
    assert np.allclose(result, expected), "Field values don't match expected"
    print("✓ Taichi fields and kernels work correctly on GPU")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Taichi GPU Configuration")
    print("="*60 + "\n")

    try:
        test_default_arch()
        test_gpu_memory_fraction()

        print("\nInitializing Taichi simulation...")
        backend = test_taichi_initialization()

        test_taichi_fields()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print(f"\nYour system is configured to use {backend.upper()} GPU backend by default.")
        print("Training will now run at full GPU speed without artificial throttling.\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
