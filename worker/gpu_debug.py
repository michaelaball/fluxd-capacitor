# gpu_debug.py
import sys
import subprocess
import platform

def print_system_info():
    print("\n===== SYSTEM INFORMATION =====")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Try to get GPU info using various methods
    try:
        if platform.system() == "Windows":
            # Try using Windows Management Instrumentation
            print("\n--- GPU Information (Windows) ---")
            gpu_info = subprocess.check_output("wmic path win32_VideoController get Name", shell=True).decode()
            print(gpu_info)
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def check_pytorch():
    print("\n===== PYTORCH STATUS =====")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                try:
                    print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
                except:
                    pass
                try:
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    print(f"  Total memory: {total_mem / 1024**3:.2f} GB")
                except:
                    pass
        else:
            # Try to figure out why CUDA isn't available
            print("\nTroubleshooting CUDA not available:")
            
            # Check if CUDA is installed
            try:
                nvcc_version = subprocess.check_output("nvcc --version", shell=True).decode()
                print(f"\nNVCC version found:\n{nvcc_version}")
            except:
                print("\nNVCC not found - CUDA toolkit might not be installed")
            
            # Check if NVIDIA driver is installed
            try:
                if platform.system() == "Windows":
                    nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
                    print(f"\nNVIDIA driver found (nvidia-smi):\n{nvidia_smi}")
                else:
                    print("\nNot on Windows - skipping nvidia-smi check")
            except:
                print("\nNVIDIA driver not found or nvidia-smi not available")
            
            # Check PyTorch CUDA status
            print("\nChecking PyTorch build information:")
            print(f"CUDA build version: {torch.version.cuda}")
            build_info = {k: v for k, v in vars(torch.backends.cuda).items() if not k.startswith('__')}
            print(f"CUDA build flags: {build_info}")
            
    except ImportError:
        print("PyTorch is not installed")

if __name__ == "__main__":
    print_system_info()
    check_pytorch()
    
    print("\n===== RECOMMENDED STEPS =====")
    print("1. Make sure you have an NVIDIA GPU")
    print("2. Install the latest NVIDIA drivers from https://www.nvidia.com/Download/index.aspx")
    print("3. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
    print("4. Reinstall PyTorch with CUDA support using:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("5. Restart your computer after installing drivers")