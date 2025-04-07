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
            # Add this code to the SDXLWorker class in worker.py

    # Add this code to the SDXLWorker class in worker.py

    def _load_lora_models(self, lora_models, lora_strengths):
        """
        Load LoRA models into the pipeline
        
        Args:
            lora_models (list): List of LoRA model names
            lora_strengths (list): List of LoRA strength values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            from pathlib import Path
            from storage import S3Storage
            
            if not hasattr(self, 'loaded_loras'):
                self.loaded_loras = {}
                
            print(f"Loading LoRA models: {lora_models} with strengths: {lora_strengths}")
            
            # Check if the same models with same strengths are already loaded
            current_loras = {model: strength for model, strength in zip(lora_models, lora_strengths)}
            if self.loaded_loras == current_loras:
                print("Same LoRA models already loaded with same strengths, skipping reload")
                return True
                
            # Initialize storage handler
            storage = S3Storage()
            
            # Define local directory for LoRA models
            lora_dir = Path("./loras")
            
            # Download LoRA models from S3
            downloaded_files = storage.download_specific_files(
                lora_models,
                "loras/",
                str(lora_dir)
            )
            
            if len(downloaded_files) != len(lora_models):
                print(f"Warning: Not all LoRA models were downloaded. Expected {len(lora_models)}, got {len(downloaded_files)}")
            
            # Unload any existing LoRA weights
            try:
                if hasattr(self.pipe, "unload_lora_weights"):
                    print("Unloading previous LoRA weights")
                    self.pipe.unload_lora_weights()
            except Exception as e:
                print(f"Error unloading LoRA weights: {e}")
            
            # Load each adapter one by one
            for i, model_name in enumerate(lora_models):
                try:
                    if not model_name.endswith('.safetensors'):
                        file_name = f"{model_name}.safetensors"
                    else:
                        file_name = model_name
                        
                    model_path = lora_dir / file_name
                    if model_path.exists():
                        strength = float(lora_strengths[i]) if i < len(lora_strengths) else 1.0
                        print(f"Loading LoRA {model_name} from {model_path} with strength {strength}")
                        
                        # The FluxPipeline may use a different API than standard diffusers
                        # Try different loading methods
                        adapter_name = Path(file_name).stem
                        
                        if hasattr(self.pipe, "load_lora_weights"):
                            self.pipe.load_lora_weights(str(model_path))
                            
                            if hasattr(self.pipe, "set_adapters"):
                                # For newer diffusers
                                self.pipe.set_adapters([adapter_name], adapter_weights=[strength])
                            elif hasattr(self.pipe, "fuse_lora"):
                                # For some versions that use fuse_lora
                                self.pipe.fuse_lora(lora_scale=strength)
                            else:
                                print(f"Loaded LoRA {model_name} but couldn't set weight {strength}")
                    else:
                        print(f"Warning: LoRA file not found: {model_path}")
                except Exception as e:
                    print(f"Error loading LoRA {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update loaded_loras dictionary
            self.loaded_loras = current_loras
            
            return True
        except Exception as e:
            print(f"Error loading LoRA models: {e}")
            import traceback
            traceback.print_exc()
            return False
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
            
            # Handle LoRA models and strengths
            lora_models = []
            lora_strengths = []
            
            if "lora_model" in job_data and job_data["lora_model"]:
                lora_models = job_data["lora_model"].split(',')
                lora_models = [model.strip() for model in lora_models]
                
                if "lora_strength" in job_data and job_data["lora_strength"]:
                    lora_strengths = job_data["lora_strength"].split(',')
                    lora_strengths = [strength.strip() for strength in lora_strengths]
                    
                    # Ensure we have a strength value for each model
                    if len(lora_strengths) < len(lora_models):
                        lora_strengths.extend(['1.0'] * (len(lora_models) - len(lora_strengths)))
            
            # Load LoRA models if specified
            if lora_models:
                lora_success = self._load_lora_models(lora_models, lora_strengths)
                if not lora_success:
                    print("Warning: Failed to load some LoRA models. Continuing with available models.")
            
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
                        "lora_model": lora_models if lora_models else None,
                        "lora_strength": lora_strengths if lora_strengths else None,
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
            import traceback
            traceback.print_exc()
            
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