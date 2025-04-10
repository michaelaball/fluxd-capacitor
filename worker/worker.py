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

# Art styles mapping
ART_STYLES = {
    "Studio Ghibli": "in the style of Studio Ghibli anime, soft watercolor textures, dreamlike pastel colors, hand-drawn animation aesthetic, intricate background details",
    "Digital Synthwave": "in the style of retrowave synthwave digital art, neon color palette, 80s cyberpunk aesthetic, vibrant purple and pink gradients, geometric landscape with retro futuristic elements",
    "Art Deco": "in the style of art deco",
    "Anime": "in the style of a Japanese anime cartoon"
}


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
                self.pipe.enable_model_cpu_offload()  # Reintroduce CPU offloading
                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                
                print("Model loaded successfully with CPU offloading.")
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
    

    def batch_jobs_with_constraints(redis_client, queue_name, max_batch_size=4):
        """
        Batch jobs from Redis with consistent parameters
        
        Args:
            redis_client (Redis): Redis client
            queue_name (str): Name of the job queue
            max_batch_size (int): Maximum number of jobs to batch
        
        Returns:
            list: Batch of jobs with consistent parameters, or empty list if no jobs
        """
        # Pull first job to set the baseline
        first_job_id = redis_client.lpop(queue_name)
        # Safe return if nothing to unpack
        if not first_job_id:
            return [], []
        
        # Fetch first job details
        first_job_data = json.loads(redis_client.get(f"job:{first_job_id}"))
        
        # Baseline parameters to match
        baseline_height = first_job_data.get("height")
        baseline_width = first_job_data.get("width")
        baseline_lora_models = first_job_data.get("lora_model")
        baseline_lora_strengths = first_job_data.get("lora_strength")
        
        # Batch to return
        batch_jobs = [first_job_data]
        batch_job_ids = [first_job_id]
        
        # Try to pull additional jobs
        while len(batch_jobs) < max_batch_size:
            # Pull next job
            next_job_id = redis_client.lpop(queue_name)
            if not next_job_id:
                break
            
            # Fetch job details
            try:
                next_job_data = json.loads(redis_client.get(f"job:{next_job_id}"))
                
                # Check consistency
                is_consistent = (
                    next_job_data.get("height") == baseline_height and
                    next_job_data.get("width") == baseline_width and
                    next_job_data.get("lora_model", '') == baseline_lora_models and
                    next_job_data.get("lora_strength", '') == baseline_lora_strengths
                )
                
                if is_consistent:
                    batch_jobs.append(next_job_data)
                    batch_job_ids.append(next_job_id)
                else:
                    # If inconsistent, return job to queue
                    redis_client.rpush(queue_name, next_job_id)
                    break
            
            except Exception as e:
                print(f"Error processing job {next_job_id}: {e}")
                # Return job to queue on error
                redis_client.rpush(queue_name, next_job_id)
                break
        
        return batch_jobs, batch_job_ids

    # Continuous job processing loop
    # Continuous job processing loop
    while True:
        try:
            # Batch jobs with consistent parameters
            batch_jobs, batch_job_ids = batch_jobs_with_constraints(redis_client, queue_name)
            if not batch_jobs:
                time.sleep(polling_interval)
                continue

            # Prepare batch prompts and other parameters
            prompts = []
            negative_prompts = []
            jobs_metadata = []

            # Prepare jobs for batch processing
            for job_data in batch_jobs:
                # Update job status
                job_data["status"] = "processing"
                job_data["device"] = device
                
                prompts.append(job_data.get("prompt", "A beautiful dreamscape"))
                negative_prompts.append(job_data.get("negative_prompt", ""))
                jobs_metadata.append(job_data)

            # Use first job's parameters for batch processing
            first_job = jobs_metadata[0]
            height = int(first_job.get("height", 1024))
            width = int(first_job.get("width", 1024))
            steps = int(first_job.get("num_inference_steps", 20))
            guidance = float(first_job.get("guidance_scale", 7.5))
            num_images = len(batch_jobs)  # One image per job

            # Set up random seed
            seed = first_job.get("seed")
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
            print(f"[GPU {gpu_index}] Batch Processing:")
            print("Batch Parameters:")
            print(f"  Height: {height}")
            print(f"  Width: {width}")
            print(f"  Steps: {steps}")
            print(f"  Guidance Scale: {guidance}")
            print(f"  Seed: {seed}")
            print("Batch Prompts:")
            for i, prompt in enumerate(prompts, 1):
                print(f"  Job {i}: {prompt}")
            # Get LoRA models and strengths from the first job
            lora_models = first_job.get("lora_model", "")
            lora_strengths = first_job.get("lora_strength", "")

            # Convert string parameters to lists if needed
            if isinstance(lora_models, str) and lora_models:
                lora_models = [lora_models]
            elif not lora_models:
                lora_models = []
                
            if isinstance(lora_strengths, str) and lora_strengths:
                lora_strengths = [lora_strengths]
            elif not lora_strengths:
                lora_strengths = []

            # Load LoRA models if specified
            if lora_models:
                print(f"[GPU {gpu_index}] Loading LoRA models: {lora_models}")
                if not worker_ctx._load_lora_models(lora_models, lora_strengths):
                    print(f"[GPU {gpu_index}] Failed to load LoRA models")
                    # Continue anyway, will use base model
                    
            # Now generate the images
            images = worker_ctx.pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=1,
                generator=generator
            ).images
            # Process and save results for each job
            for job_id, job_data, image in zip(batch_job_ids, jobs_metadata, images):
                # Upload images
                image_urls = worker_ctx.storage.upload_images([image], job_id)

                # Prepare base64 images if configured
                base64_images = None
                if worker_ctx.storage.should_include_base64():
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    base64_images = [base64.b64encode(buffered.getvalue()).decode()]

                # Prepare job completion data
                generation_time = time.time() - start_time
                job_data.update({
                    "status": "success",
                    "output": image_urls,
                    "data": {
                        "image_urls": image_urls,
                        "base64_images": base64_images,
                        "parameters": {
                            "prompt": job_data.get("prompt"),
                            "negative_prompt": job_data.get("negative_prompt", ""),
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

            # Clear CUDA cache between batches
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {gpu_index}] Error processing batch: {e}")
            # Update job statuses to failed
            for job_id, job_data in zip(batch_job_ids, batch_jobs):
                try:
                    job_data["status"] = "error"
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