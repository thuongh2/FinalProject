import gdown
import zipfile
import os

# Function to download a file from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# Function to unzip a file
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Example usage
if __name__ == "__main__":
    # Replace with your own file ID and destination path
    google_drive_file_id = '1UAzmFz958xJrLyg2b7GyZZ1q_YJMjAQ8'
    destination_zip_path = 'file_model.zip'
    extract_to_directory = ''

    # Download the file from Google Drive
    download_file_from_google_drive(google_drive_file_id, destination_zip_path)

    # Unzip the file
    unzip_file(destination_zip_path, extract_to_directory)

    # Optional: Clean up by removing the zip file after extraction
    os.remove(destination_zip_path)

    print(f"File downloaded and extracted to '{extract_to_directory}'")
