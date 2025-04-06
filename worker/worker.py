# worker.py
import os
import json
import time
import base64
from io import BytesIO
import redis
import torch
from diffusers import StableDiffusionXLPipeline
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='SDXL Worker')
# parser.add_argument('--redis-host', default='localhost', help='Redis host')
# parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
# parser.add_argument('--redis-password', default='', help='Redis password')
parser.add_argument('--queue-name', default='sdxl_jobs', help='Redis queue name')
parser.add_argument('--polling-interval', type=float, default=1.0, help='Polling interval in seconds')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
args = parser.parse_args()

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=os.getenv('REDIS_PORT', 6379),
    password=os.getenv('REDIS_PASSWORD', ''),
    decode_responses=True  # Automatically decode bytes to str
)

# Queue name
QUEUE_NAME = args.queue_name

# Initialize the SDXL model
def initialize_model():
    print("Initializing SDXL model...")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the SDXL pipeline
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    
    # Optimize for memory if using GPU
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    print("Model initialized successfully")
    return pipe


# Process a single job with abstracted S3 upload
def process_job(job_id, pipe):
    try:
        print(f"Processing job {job_id}")
        
        # Import S3Storage class
        from storage import S3Storage
        import base64
        from io import BytesIO
        
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
        redis_client.set(f"job:{job_id}", json.dumps(job_data))
        
        # Extract parameters with defaults
        prompt = job_data.get("prompt", "")
        negative_prompt = job_data.get("negative_prompt", None)
        height = job_data.get("height", 1024)
        width = job_data.get("width", 1024)
        num_inference_steps = job_data.get("num_inference_steps", 1)
        guidance_scale = job_data.get("guidance_scale", 7.5)
        num_images = min(job_data.get("num_images", 1), 4)  # Limit to 4 images max
        seed = job_data.get("seed", int(time.time()))
        
        # Record start time
        start_time = time.time()
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        # Generate the image(s)
        images = pipe(
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
        print(f"Generated {len(images)} images in {generation_time:.2f}s")
        
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
                "generation_time": generation_time
            },
            "completedAt": time.time()
        })
        
        # Save updated job data back to Redis
        redis_client.set(f"job:{job_id}", json.dumps(job_data))
        print(f"Job {job_id} completed successfully - Images uploaded to S3")
        
        # Clean up
        storage.cleanup()
        
        return True
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        
        try:
            # Update job status to failed
            job_data = json.loads(redis_client.get(f"job:{job_id}"))
            job_data.update({
                "status": "failed",
                "error": str(e),
                "completedAt": time.time()
            })
            redis_client.set(f"job:{job_id}", json.dumps(job_data))
        except Exception as update_error:
            print(f"Error updating job status: {update_error}")
        
        return False

# Main worker loop
def worker_loop(pipe):
    print(f"Starting worker loop, polling interval: {args.polling_interval}s")
    
    while True:
        try:
            # Get a batch of jobs from the queue
            job_ids = []
            for _ in range(args.batch_size):
                job_id = redis_client.lpop(QUEUE_NAME)
                if job_id:
                    job_ids.append(job_id)
                else:
                    break
            
            if not job_ids:
                # No jobs in queue, wait before checking again
                time.sleep(args.polling_interval)
                continue
            
            print(f"Processing batch of {len(job_ids)} jobs")
            
            # Process each job in the batch
            for job_id in job_ids:
                process_job(job_id, pipe)
            
        except Exception as e:
            print(f"Error in worker loop: {e}")
            time.sleep(args.polling_interval)

# Main function
def main():
    try:
        # Test Redis connection
        redis_client.ping()
        print("Connected to Redis successfully")
        
        # Initialize the model
        pipe = initialize_model()
        
        # Start the worker loop
        worker_loop(pipe)
    except redis.ConnectionError:
        print("Failed to connect to Redis. Check your connection settings.")
    except Exception as e:
        print(f"Unhandled exception: {e}")

if __name__ == "__main__":
    main()