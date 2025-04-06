# storage.py
import os
import time
from pathlib import Path
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

class S3Storage:
    """
    Handles storage operations with S3 compatible services.
    Loads configuration from environment variables.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # S3 configuration
        self.access_key = os.getenv('S3_ACCESS_KEY')
        self.secret_key = os.getenv('S3_SECRET_KEY')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('S3_REGION', 'us-east-1')
        self.bucket_path = os.getenv('S3_BUCKET_PATH')
        self.endpoint = os.getenv('S3_ENDPOINT')
        self.include_base64 = os.getenv('INCLUDE_BASE64', 'false').lower() == 'true'
        
        # Ensure required configuration is available
        if not all([self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError("Missing required S3 configuration in .env file")
            
        # Initialize S3 client
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            endpoint_url=self.endpoint
        )
        
        # Create temp directory for file operations
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)
        
    def upload_image(self, image, job_id, index=0):
        """
        Upload a PIL Image to S3 and return the URL.
        
        Args:
            image: PIL Image object
            job_id: Unique identifier for the job
            index: Image index number
            
        Returns:
            str: URL to the uploaded image
        """
        try:
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{job_id}_{timestamp}_{index}.png"
            local_path = self.temp_dir / filename
            
            # Save image locally first
            image.save(local_path, format="PNG")
            
            # Upload to S3
            s3_key = f"{self.bucket_path}/{filename}"
            
            # Check if we should use public-read ACL or pre-signed URLs
            use_presigned = os.getenv('USE_PRESIGNED_URLS', 'true').lower() == 'true'
            
            if use_presigned:
                # Upload with private ACL if we're using pre-signed URLs
                self.client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'image/png'}
                )
                
                # Generate pre-signed URL
                url = self.generate_presigned_url(s3_key)
            else:
                # Upload with public-read ACL
                self.client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
                )
                
                # Generate public URL
                if self.endpoint:
                    # For custom S3 endpoints
                    url = f"{self.endpoint}/{self.bucket_name}/{s3_key}"
                else:
                    # For AWS S3
                    url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            # Clean up local file
            local_path.unlink()
            
            return url
        
        except Exception as e:
            print(f"Error uploading image to S3: {e}")
            # Don't delete local file in case of error for debugging
            raise
    
    def upload_images(self, images, job_id):
        """
        Upload multiple images to S3 and return URLs.
        
        Args:
            images: List of PIL Image objects
            job_id: Unique identifier for the job
            
        Returns:
            list: URLs to the uploaded images
        """
        image_urls = []
        for i, image in enumerate(images):
            url = self.upload_image(image, job_id, i)
            image_urls.append(url)
        
        return image_urls
    
    def cleanup(self):
        """Clean up temp directory if empty"""
        try:
            self.temp_dir.rmdir()
        except:
            pass
    
    def should_include_base64(self):
        """Check if base64 encoding should be included"""
        return self.include_base64