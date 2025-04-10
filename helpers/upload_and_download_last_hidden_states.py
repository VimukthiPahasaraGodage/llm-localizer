import os
import zipfile
import boto3

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION = 'us-east-1'
BUCKET_NAME = 'final-fyp-data'

def zip_and_upload_to_s3(local_dir, bucket_name):
    # Normalize path
    local_dir = os.path.normpath(local_dir)
    # Create a zip file in a temporary location
    zip_filename = os.path.basename(local_dir) + '.zip'
    zip_path = os.path.join('/tmp', zip_filename)

    # Zip the contents of the folder
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, os.path.dirname(local_dir))
                zipf.write(full_path, arcname)

    # Upload to S3 with the same path structure
    s3_key = os.path.normpath(local_dir) + '.zip'
    s3_key = s3_key.replace("\\", "/")  # Ensure S3 key uses forward slashes

    s3 = boto3.client(
        's3',
        aws_access_key_id="",
        aws_secret_access_key="",
        region_name=AWS_REGION
    )
    s3.upload_file(zip_path, bucket_name, s3_key)

    print(f"Uploaded {zip_path} to s3://{bucket_name}/{s3_key}")

def download_and_unzip(s3_key, local_file_path, bucket_name):
    # bucket_name = 'final-fyp-data'
    # s3_key = 'data/tensors/defects4j/v2/3.zip'
    # local_file_path = 'data/tensors/defects4j/v2/3.zip'

    # Create the local directory if it doesn't exist
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print(f"Downloading ...")
    s3.download_file(bucket_name, s3_key, local_file_path)

    print(f"Downloaded to {local_file_path}")

    print(f"Unzipping ...")
    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(local_file_path))
    print(f"Unzipping completed!")

    # Set the directory path where the zip was extracted
    dir_path = os.path.dirname(local_file_path)  # Example: 'data/tensors/defects4j/v2'

    # Walk through the directory and list all files
    file_count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_count += 1
    print("File count: ", str(file_count))


# zip_and_upload_to_s3('data/tensors/defetcts4j/v2/3/', BUCKET_NAME)
# download_and_unzip('data/tensors/defetcts4j/v2/3.zip', 'data/tensors/defects4j/v2/3.zip', BUCKET_NAME)
