# Fix 1: Complete memory cleanup function
def cleanup_gpu_memory(self):
    """Thoroughly clean up GPU memory between generations"""
    if self.gpu_index >= 0:
        try:
            # Standard cache clearing
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Move pipe to CPU temporarily if needed
            if hasattr(self, 'pipe') and self.pipe is not None:
                self.pipe = self.pipe.to('cpu')
                torch.cuda.empty_cache()
                gc.collect()
                self.pipe = self.pipe.to(self.device)
            
            print(f"Thorough GPU memory cleanup completed on {self.device}")
            
            # Print memory usage for debugging
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated = torch.cuda.memory_allocated(self.gpu_index) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.gpu_index) / (1024**3)
                print(f"GPU {self.gpu_index} memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except Exception as e:
            print(f"Error during thorough GPU cleanup: {e}")

# Fix 2: Improved LoRA management function
def _manage_lora_weights(self, new_loras=None, new_strengths=None):
    """
    Intelligently manage LoRA weights - only unload if necessary
    
    Args:
        new_loras: List of new LoRA models to load, or None if no new LoRAs
        new_strengths: List of corresponding strengths for new LoRAs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If no new LoRAs are specified and no LoRAs are currently loaded, do nothing
        if not new_loras and not self.loaded_loras:
            return True
            
        # If new LoRAs are different from current ones, unload current LoRAs
        current_loras = {model: strength for model, strength in 
                         zip(new_loras or [], new_strengths or [])}
        
        if current_loras != self.loaded_loras:
            # Only unload if we need to change LoRAs
            print("LoRA configuration changed, unloading current LoRAs")
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
                
                # Additional cleanup for adapters
                if hasattr(self.pipe, "text_encoder") and hasattr(self.pipe.text_encoder, "delete_adapters"):
                    self.pipe.text_encoder.delete_adapters()
                if hasattr(self.pipe, "unet") and hasattr(self.pipe.unet, "delete_adapters"):
                    self.pipe.unet.delete_adapters()
                
                # Reset loaded_loras tracking only if we actually unloaded
                self.loaded_loras = {}
                print("Previous LoRA weights unloaded")
        else:
            # LoRAs are the same, keep them loaded
            print("Keeping current LoRAs loaded (configuration unchanged)")
            return True
            
    except Exception as e:
        print(f"Error managing LoRA weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

# Fix 3: Better model initialization with memory-efficient settings
def initialize_model(self):
    """Initialize the FLUX.1-dev model on the specific GPU with memory optimizations"""
    try:
        from diffusers import FluxPipeline
        import torch
        
        # Set device
        if self.gpu_index >= 0:
            torch.cuda.set_device(self.gpu_index)
        
        print(f"Using device: {self.device}")
        
        # Load the FLUX pipeline
        model_id = "black-forest-labs/FLUX.1-dev"
        
        # Determine torch dtype based on device
        if self.gpu_index >= 0:
            # Use bfloat16 as specified
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
            
        # Memory optimization settings
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="bf16", # Ensure we get the optimized variant if available
            use_safetensors=True
        )
        
        # Apply memory optimizations
        if self.gpu_index >= 0:
            self.pipe.enable_model_cpu_offload()
            
            # Enable attention slicing to reduce memory usage
            self.pipe.enable_attention_slicing()
            
            # Enable vae slicing for memory efficiency
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
                
            # Enable xformers memory efficient attention if available
            try:
                import xformers
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except ImportError:
                print("xformers not available, skipping memory efficient attention")
        else:
            self.pipe = self.pipe.to(self.device)
        
        print(f"FLUX.1-dev model initialized successfully on {self.device} with memory optimizations")
        return True
        
    except Exception as e:
        print(f"Error initializing model on {self.device}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Fix 4: Modified process_job to integrate memory cleanup
def process_job(self, job_id):
    """Process a single job with better memory management"""
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
        
        # Extract parameters with defaults
        prompt = job_data.get("prompt", "")
        negative_prompt = job_data.get("negative_prompt", "")
        
        # Thorough cleanup BEFORE loading LoRAs
        self.cleanup_gpu_memory()
        
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
        
        # Intelligently manage LoRA weights - only unload if needed
        self._manage_lora_weights(lora_models, lora_strengths)
        
        # Load LoRA models if specified
        if lora_models:
            # Check if we need to load - either we have no LoRAs loaded, or different ones
            current_loras = {model: strength for model, strength in 
                             zip(lora_models, lora_strengths)}
            
            if current_loras != self.loaded_loras:
                print(f"Loading new LoRA configuration: {current_loras}")
                lora_success = self._load_lora_models(lora_models, lora_strengths)
                if not lora_success:
                    print("Warning: Failed to load some LoRA models. Continuing with available models.")
            else:
                print("Reusing already loaded LoRAs with same configuration")
        
        # Process other parameters
        # ... (rest of your existing parameter parsing code)
            
        # Record start time
        start_time = time.time()
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate the image(s) with memory optimization
        try:
            # Thorough cleanup before generation
            self.cleanup_gpu_memory()
                
            # Generate with additional memory-saving options
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
            
            # Thorough cleanup after generation
            self.cleanup_gpu_memory()
            
        except torch.cuda.OutOfMemoryError as oom:
            print(f"Out of memory error while generating images: {oom}")
            # Try to recover with extra aggressive memory cleanup
            self.cleanup_gpu_memory()
            self._unload_all_loras()
            # Propagate the error
            raise
        
        # Continue with the rest of your function...
        # (image uploads, job updates, etc.)
        
        # Final cleanup before returning
        self.cleanup_gpu_memory()
        # Do NOT unload LoRAs here - we want to keep them for the next job if it uses the same ones
        
        return True
            
    except Exception as e:
        print(f"Error processing job {job_id} on {self.device}: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup even after errors
        try:
            self.cleanup_gpu_memory()
            # Do NOT unload LoRAs here - we want to keep them for potential reuse
        except:
            pass
            
        # Update job status code...
        # (your existing error handling code)
        
        return False