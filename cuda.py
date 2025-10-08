import torch

print("=" * 60)
print("CUDA Configuration Check")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_memory:.2f} GB")
    
    try:
        x = torch.rand(5, 3).cuda()
        print(f"\nCUDA is working! Test tensor created on GPU")
        print(f"Test tensor device: {x.device}")
    except Exception as e:
        print(f"\nError creating tensor on GPU: {e}")
else:
    print("\nCUDA is not available")
    print("Check your CUDA installation and drivers")