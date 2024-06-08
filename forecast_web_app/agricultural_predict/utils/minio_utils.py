from minio import Minio

BUCKET_NAME = 'test'
MINIO_URL = 'v270vdxl-9000.asse.devtunnels.ms'
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET = 'minio123'
PATH = "./file/"

def upload_object(filename, data, length):
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)

    found = client.bucket_exists(BUCKET_NAME)
    if not found:
        client.make_bucket(BUCKET_NAME)
    else:
        print(f"Bucket {BUCKET_NAME} already exists")

    file = client.put_object(BUCKET_NAME, filename, data, length)
    print(file.object_name)
    print(file.version_id)
    print(file.etag)
    print(f"{filename} is successfully uploaded to bucket {BUCKET_NAME}.")
    return file


def fupload_object(filename, data, length= None):
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)

    # Make bucket if not exist.
    found = client.bucket_exists(BUCKET_NAME)
    if not found:
        client.make_bucket(BUCKET_NAME)
    else:
        print(f"Bucket {BUCKET_NAME} already exists")

    file = client.fput_object(BUCKET_NAME, filename, data)
    print(f"{filename} is successfully uploaded to bucket {BUCKET_NAME}.")
    return file

def get_minio_object(filename, bucket= BUCKET_NAME):
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)
    client.fget_object(bucket, filename, PATH + filename)
    return PATH + filename