# worker.py
import os
import json
import time
import base64
from io import BytesIO
import redis
import torch
import threading
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='SDXL Worker')
parser.add_argument('--queue-name', default='sdxl_jobs', help='Redis queue name')
parser.add_argument('--polling-interval', type=float, default=1.0, help='Polling interval in seconds')
args = parser.parse_args()

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', ''),
    decode_responses=True  # Automatically decode bytes to str
)

# Queue name
QUEUE_NAME = args.queue_name

# Redis lock for synchronization
REDIS_LOCK_KEY = f"{QUEUE_NAME}:lock"

# Detect available GPUs
def get_available_gpus():
    """Detect and return available GPUs with their memory info"""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available. Using CPU.")
        return []
    
    gpu_count = torch.cuda.device_count()
    available_gpus = []
    
    for i in range(gpu_count):
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
            
            # Check if device is usable
            torch.cuda.set_device(i)
            test_tensor = torch.zeros((1, 1)).cuda()
            test_tensor + 1  # Simple operation to test the device
            
            # Get free memory
            free_memory = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # Convert to GB
            
            available_gpus.append({
                'index': i,
                'name': props.name,
                'total_memory_gb': total_memory,
                'free_memory_gb': free_memory
            })
            
            print(f"GPU {i}: {props.name}, {free_memory:.2f}GB free / {total_memory:.2f}GB total")
            
        except Exception as e:
            print(f"GPU {i} is not usable: {e}")
    
    return available_gpus

# Updated SDXLWorker class with separated initialization
class SDXLWorker(threading.Thread):
    def __init__(self, gpu_index):
        threading.Thread.__init__(self)
        self.gpu_index = gpu_index
        self.device = f"cuda:{gpu_index}" if gpu_index >= 0 else "cpu"
        self.pipe = None
        self.daemon = True  # Thread will exit when main program exits
        self.running = True
    
    def run(self):
        """Main worker thread that processes jobs"""
        # Model is already initialized before thread starts
        print(f"Worker thread started for {self.device}")
        
        while self.running:
            try:
                # Pull directly from Redis queue
                job_id = redis_client.lpop(QUEUE_NAME)
                
                # If no job, wait and try again
                if not job_id:
                    time.sleep(args.polling_interval)
                    continue
                
                # Process the job
                self.process_job(job_id)
                
                # Clear CUDA cache after processing
                if self.gpu_index >= 0:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        print(f"Cleared CUDA cache on {self.device}")
                        
                        # Print memory usage for debugging
                        if hasattr(torch.cuda, 'memory_allocated'):
                            allocated = torch.cuda.memory_allocated(self.gpu_index) / (1024**3)
                            reserved = torch.cuda.memory_reserved(self.gpu_index) / (1024**3)
                            print(f"GPU {self.gpu_index} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    except Exception as e:
                        print(f"Error clearing CUDA cache: {e}")
                
            except Exception as e:
                print(f"Error in worker thread for device {self.device}: {e}")
                import traceback
                traceback.print_exc()
                # Brief delay before continuing
                time.sleep(1)
    
    def stop(self):
        """Signal the worker to stop"""
        self.running = True

# Modified main function for sequential GPU initialization
def main():
    try:
        # Test Redis connection
        redis_client.ping()
        print("Connected to Redis successfully")
        
        # Detect available GPUs
        gpus = get_available_gpus()
        
        # Print summary
        if gpus:
            print(f"Found {len(gpus)} usable GPUs")
        else:
            print("No usable GPUs found, will use CPU")
            gpus = [{'index': -1, 'name': 'CPU'}]  # Add CPU as a fallback
        
        # Create worker objects but don't start them yet
        workers = []
        for gpu in gpus:
            worker = SDXLWorker(gpu['index'])
            workers.append(worker)
        
        # Initialize each worker sequentially to avoid resource conflicts
        print("Initializing workers sequentially...")
        for i, worker in enumerate(workers):
            print(f"Initializing worker {i+1}/{len(workers)} (Device: {worker.device})")
            # Initialize model without starting the thread
            success = worker.initialize_model()
            if not success:
                print(f"WARNING: Failed to initialize worker for {worker.device}")
        
        # Now start all worker threads
        print("Starting worker threads...")
        for worker in workers:
            if hasattr(worker, 'pipe') and worker.pipe is not None:
                worker.start()
                print(f"Started worker thread for {worker.device}")
            else:
                print(f"Skipping worker thread for {worker.device} due to initialization failure")
        
        print(f"Started {sum(1 for w in workers if hasattr(w, 'pipe') and w.pipe is not None)} worker threads")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Received keyboard interrupt, shutting down workers...")
            for worker in workers:
                worker.stop()
            
    except redis.ConnectionError:
        print("Failed to connect to Redis. Check your connection settings.")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()