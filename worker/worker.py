import os
import json
import time
import base64
from io import BytesIO
import redis
import torch
import argparse
import multiprocessing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser(description='SDXL Worker')
parser.add_argument('--queue-name', default='sdxl_jobs')
parser.add_argument('--polling-interval', type=float, default=1.0)
args = parser.parse_args()

def create_redis_client():
    """Create a new Redis client"""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', ''),
        decode_responses=True
    )

def get_available_gpus():
    """Detect available GPUs"""
    gpus = []
    if not torch.cuda.is_available():
        return gpus
    
    for i in range(torch.cuda.device_count()):
        try:
            # Verify GPU is usable
            props = torch.cuda.get_device_properties(i)
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
            gpus.append({'index': i, 'name': props.name, 'free_memory_gb': free_mem})
        except Exception:
            continue
    return gpus

def run_worker(gpu_index, queue_name, polling_interval):
    """
    Worker process for a specific GPU
    
    Args:
        gpu_index (int): Index of the GPU to use
        queue_name (str): Redis queue to pull jobs from
        polling_interval (float): Time to wait between job checks
    """
    # Strictly limit this process to ONLY see its assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = "cuda:0"  # Remapped to this GPU due to CUDA_VISIBLE_DEVICES

    # Create a separate Redis client for this process
    redis_client = create_redis_client()

    print(f"[GPU {gpu_index}] Worker started on {device}")

    # Importing inside the function to ensure proper process isolation
    from diffusers import FluxPipeline
    from storage import S3Storage
    from pathlib import Path

    class WorkerContext:
        def __init__(self):
            self.pipe = None
            self.loaded_loras = {}
            self.storage = S3Storage()

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
                # Track loaded LoRAs
                current_loras = {model: strength for model, strength in zip(lora_models, lora_strengths)}
                
                # Skip if same LoRAs are already loaded
                if self.loaded_loras == current_loras:
                    print("Same LoRA models already loaded with same strengths, skipping reload")
                    return True
                
                # Define local directory for LoRA models
                lora_dir = Path("./loras")
                lora_dir.mkdir(exist_ok=True)
                
                # Download LoRA models from S3
                downloaded_files = self.storage.download_specific_files(
                    lora_models,
                    "loras/",
                    str(lora_dir)
                )
                
                if len(downloaded_files) != len(lora_models):
                    print(f"Warning: Not all LoRA models were downloaded. Expected {len(lora_models)}, got {len(downloaded_files)}")
                
                # Unload any existing LoRA weights
                if hasattr(self.pipe, "unload_lora_weights"):
                    print("Unloading previous LoRA weights")
                    self.pipe.unload_lora_weights()
                
                # Load each LoRA
                for i, model_name in enumerate(lora_models):
                    if not model_name.endswith('.safetensors'):
                        file_name = f"{model_name}.safetensors"
                    else:
                        file_name = model_name
                        
                    model_path = lora_dir / file_name
                    if model_path.exists():
                        strength = float(lora_strengths[i]) if i < len(lora_strengths) else 1.0
                        print(f"Loading LoRA {model_name} with strength {strength}")
                        
                        try:
                            # Load the LoRA weights
                            self.pipe.load_lora_weights(
                                str(model_path),
                                cross_attention_kwargs={"scale": strength}
                            )
                        except Exception as load_error:
                            print(f"Error loading LoRA {model_name}: {load_error}")
                            continue
                    else:
                        print(f"Warning: LoRA file not found: {model_path}")
                
                # Update loaded_loras dictionary
                self.loaded_loras = current_loras
                print("LoRA loading complete")
                return True
                
            except Exception as e:
                print(f"Error loading LoRA models: {e}")
                import traceback
                traceback.print_exc()
                return False

        def initialize_model(self):
            """Initialize the FLUX.1-dev model with memory optimizations"""
            try:
                # Load pipeline
                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.bfloat16
                )
                
                # Memory optimization techniques
                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                
                print("Model loaded successfully.")
                return True
                
            except Exception as e:
                print(f"Model loading failed: {e}")
                return False

    # Initialize worker context
    worker_ctx = WorkerContext()

    # Load model at startup
    if not worker_ctx.initialize_model():
        print(f"[GPU {gpu_index}] Failed to initialize model. Exiting.")
        return

    # Continuous job processing loop
    while True:
        try:
            # Pull job from Redis queue
            job_id = redis_client.lpop(queue_name)
            if not job_id:
                time.sleep(polling_interval)
                continue

            # Fetch job details
            job_data_raw = redis_client.get(f"job:{job_id}")
            if not job_data_raw:
                print(f"[GPU {gpu_index}] Job {job_id} not found")
                continue

            job_data = json.loads(job_data_raw)
            
            # Update job status
            job_data["status"] = "processing"
            job_data["device"] = device
            redis_client.set(f"job:{job_id}", json.dumps(job_data))

            # Handle LoRA models
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
                lora_success = worker_ctx._load_lora_models(lora_models, lora_strengths)
                if not lora_success:
                    print("Warning: Failed to load some LoRA models. Continuing with available models.")

            # Extract job parameters with sensible defaults
            prompt = job_data.get("prompt", "A beautiful dreamscape")
            negative_prompt = job_data.get("negative_prompt", "")
            height = int(job_data.get("height", 1024))
            width = int(job_data.get("width", 1024))
            steps = int(job_data.get("num_inference_steps", 20))
            guidance = float(job_data.get("guidance_scale", 7.5))
            num_images = min(int(job_data.get("num_images", 1)), 4)

            # Set up random seed
            seed = job_data.get("seed")
            if seed is None or seed == "null":
                seed = int(time.time())
            else:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    seed = int(time.time())

            generator = torch.Generator(device).manual_seed(seed)

            # Image generation
            start_time = time.time()
            images = worker_ctx.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=num_images,
                generator=generator
            ).images

            # Upload images
            image_urls = worker_ctx.storage.upload_images(images, job_id)

            # Prepare base64 images if configured
            base64_images = None
            if worker_ctx.storage.should_include_base64():
                base64_images = []
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    base64_images.append(img_str)

            # Prepare job completion data
            generation_time = time.time() - start_time
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
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed
                    },
                    "generation_time": generation_time
                },
                "completedAt": time.time()
            })

            # Save completed job data
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
            
            print(f"[GPU {gpu_index}] Job {job_id} completed in {generation_time:.2f}s.")

            # Clear CUDA cache between jobs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {gpu_index}] Error processing job: {e}")
            # Update job status to failed
            try:
                job_data["status"] = "failed"
                job_data["error"] = str(e)
                redis_client.set(f"job:{job_id}", json.dumps(job_data))
            except Exception:
                pass
            time.sleep(2)

def main():
    try:
        # Set multiprocessing start method to spawn
        multiprocessing.set_start_method('spawn')

        # Verify Redis connection
        redis_client = create_redis_client()
        redis_client.ping()
        print("Connected to Redis")

        # Detect available GPUs
        gpus = get_available_gpus()
        print(f"Found {len(gpus)} usable GPUs")

        # Spawn a process for each GPU
        processes = []
        for gpu in gpus:
            p = multiprocessing.Process(
                target=run_worker,
                args=(gpu['index'], args.queue_name, args.polling_interval),
            )
            p.start()
            processes.append(p)
            time.sleep(1)  # Stagger startup to avoid load spikes

        # Wait for all processes
        for p in processes:
            p.join()

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()