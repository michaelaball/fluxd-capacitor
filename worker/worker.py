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

# Worker class for GPU-specific processing
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
        # Initialize model on the specific GPU
        print(f"Initializing model on {self.device}")
        self.initialize_model()
        
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
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in worker thread for device {self.device}: {e}")
                # Brief delay before continuing
                time.sleep(1)
    
    def stop(self):
        """Signal the worker to stop"""
        self.running = False

    def initialize_model(self):
        """Initialize the FLUX.1-dev model on the specific GPU"""
        try:
            from diffusers import FluxPipeline
            import torch
            
            # Set device
            if self.gpu_index >= 0:
                torch.cuda.set_device(self.gpu_index)
            
            print(f"Using device: {self.device}")
            
            # Load the FLUX pipeline (hardcoded)
            model_id = "black-forest-labs/FLUX.1-dev"
            
            # Determine torch dtype based on device
            if self.gpu_index >= 0:
                # Use bfloat16 as specified
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
                
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
            
            # Set the model to device
            if self.gpu_index >= 0:
                self.pipe.enable_model_cpu_offload()  # Offload to save VRAM
            else:
                self.pipe = self.pipe.to(self.device)
            
            print(f"FLUX.1-dev model initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model on {self.device}: {e}")
            raise

    def process_job(self, job_id):
        """Process a single job"""
        try:
            print(f"Processing job {job_id} on {self.device}")
            
            # Import S3Storage class
            from storage import S3Storage
            
            # Initialize storage handler
            storage = S3Storage()
            
            # Get job data from Redis
            job_data_raw = redis_client.get(f"job:{job_id}")
            if not job_data_raw:
                print(f"Job {job_id} not found in Redis")
                return False
            
            # Parse job data
            job_data = json.loads(job_data_raw)
            
            # Update job status to processing
            job_data["status"] = "processing"
            job_data["device"] = self.device
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
            
            # Extract parameters with defaults, handling string inputs
            prompt = job_data.get("prompt", "")
            negative_prompt = job_data.get("negative_prompt", "")
            
            # Handle height/width as strings or integers
            try:
                height = int(job_data.get("height", 1024))
            except (TypeError, ValueError):
                height = 1024
                
            try:
                width = int(job_data.get("width", 1024))
            except (TypeError, ValueError):
                width = 1024
                
            # Handle num_inference_steps as string or integer
            try:
                num_inference_steps = int(job_data.get("num_inference_steps", 20))
            except (TypeError, ValueError):
                num_inference_steps = 20
                
            # Handle guidance_scale as string or float
            try:
                guidance_scale = float(job_data.get("guidance_scale", 7.5))
            except (TypeError, ValueError):
                guidance_scale = 7.5
                
            # Get number of images
            num_images = min(job_data.get("num_images", 1), 4)  # Limit to 4 images max
            
            # Handle seed (null means random)
            seed = job_data.get("seed")
            if seed is None or seed == "null":
                seed = int(time.time())
            else:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    seed = int(time.time())
            
            # Record start time
            start_time = time.time()
            
            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate the image(s)
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator
            ).images
            
            # Calculate generation time
            generation_time = time.time() - start_time
            print(f"Generated {len(images)} images in {generation_time:.2f}s on {self.device}")
            
            # Upload images to S3
            image_urls = storage.upload_images(images, job_id)
            
            # If configured to include base64, prepare those too
            base64_images = None
            if storage.should_include_base64():
                base64_images = []
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    base64_images.append(img_str)
            
            # Update job with result
            job_data.update({
                "status": "completed",
                "result": {
                    "image_urls": image_urls,
                    "base64_images": base64_images,
                    "parameters": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                    },
                    "generation_time": generation_time,
                    "device": self.device
                },
                "completedAt": time.time()
            })
            
            # Save updated job data back to Redis
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
            print(f"Job {job_id} completed successfully on {self.device}")
            
            # Clean up
            storage.cleanup()
            
            return True
            
        except Exception as e:
            print(f"Error processing job {job_id} on {self.device}: {e}")
            
            try:
                # Update job status to failed
                job_data = json.loads(redis_client.get(f"job:{job_id}"))
                job_data.update({
                    "status": "failed",
                    "error": str(e),
                    "device": self.device,
                    "completedAt": time.time()
                })
                redis_client.set(f"job:{job_id}", json.dumps(job_data))
            except Exception as update_error:
                print(f"Error updating job status: {update_error}")
            
            return False

# Main function
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
        
        # Start worker threads for each GPU
        workers = []
        for gpu in gpus:
            worker = SDXLWorker(gpu['index'])
            worker.start()
            workers.append(worker)
        
        print(f"Started {len(workers)} worker threads")
        
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

if __name__ == "__main__":
    main()