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
        

        def download_from_s3(self, s3_prefix, local_dir, force_download=False):
            """
            Download files from S3 bucket with a specific prefix to a local directory
            
            Args:
                s3_prefix (str): Prefix in S3 bucket to download from (e.g., 'loras/')
                local_dir (str): Local directory to download files to
                force_download (bool): If True, download even if file exists locally
                
            Returns:
                list: List of downloaded file paths
            """
            import os
            from pathlib import Path
            
            # Ensure the local directory exists
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # List objects with the given prefix
            try:
                paginator = self.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
                
                downloaded_files = []
                
                for page in pages:
                    if 'Contents' not in page:
                        print(f"No files found with prefix '{s3_prefix}' in bucket '{self.bucket_name}'")
                        return downloaded_files
                        
                    for obj in page['Contents']:
                        # Get the file key and name
                        file_key = obj['Key']
                        file_name = os.path.basename(file_key)
                        
                        if not file_name:  # Skip if it's a directory
                            continue
                            
                        local_file_path = local_path / file_name
                        
                        # Check if we need to download the file
                        if force_download or not local_file_path.exists():
                            print(f"Downloading {file_key} to {local_file_path}")
                            
                            # Ensure sub-directories exist
                            local_file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Download the file
                            self.client.download_file(
                                self.bucket_name,
                                file_key,
                                str(local_file_path)
                            )
                            
                            downloaded_files.append(str(local_file_path))
                        else:
                            print(f"File already exists: {local_file_path}")
                            downloaded_files.append(str(local_file_path))
                            
                return downloaded_files
                
            except Exception as e:
                print(f"Error downloading files from S3: {e}")
                return []

        def download_specific_files(self, file_names, s3_prefix, local_dir, force_download=False):
            """
            Download specific files from S3 bucket
            
            Args:
                file_names (list): List of file names to download
                s3_prefix (str): Prefix in S3 bucket to download from (e.g., 'loras/')
                local_dir (str): Local directory to download files to
                force_download (bool): If True, download even if file exists locally
                
            Returns:
                list: List of downloaded file paths
            """
            import os
            from pathlib import Path
            
            # Ensure the local directory exists
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            
            downloaded_files = []
            
            for file_name in file_names:
                # Ensure file has safetensors extension if not already
                if not file_name.endswith('.safetensors'):
                    file_name = f"{file_name}.safetensors"
                    
                # Create S3 key and local path
                file_key = f"{s3_prefix.rstrip('/')}/{file_name}"
                local_file_path = local_path / file_name
                
                # Check if we need to download the file
                if force_download or not local_file_path.exists():
                    try:
                        print(f"Downloading {file_key} to {local_file_path}")
                        
                        # Download the file
                        self.client.download_file(
                            self.bucket_name,
                            file_key,
                            str(local_file_path)
                        )
                        
                        downloaded_files.append(str(local_file_path))
                    except Exception as e:
                        print(f"Error downloading {file_key}: {e}")
                else:
                    print(f"File already exists: {local_file_path}")
                    downloaded_files.append(str(local_file_path))
                        
            return downloaded_files 
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
            s3_key = f"images/{filename}"
            
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
        
    def generate_presigned_url(self, s3_key):
        """
        Generate a pre-signed URL for an S3 object with configurable expiration time
        
        Args:
            s3_key: The key of the object in S3
            
        Returns:
            str: A pre-signed URL with time-limited access
        """
        # Get expiration time from environment variable (default: 24 hours)
        expiry_seconds = int(os.getenv('URL_EXPIRATION_SECONDS', 86400))
        
        try:
            # Generate the pre-signed URL
            response = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiry_seconds
            )
            return response
        except ClientError as e:
            print(f"Error generating pre-signed URL: {e}")
            raise

    
    # Add this function to your storage.py file

