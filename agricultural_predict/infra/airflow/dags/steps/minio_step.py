from minio import Minio

BUCKET_NAME = 'final-project'
MINIO_URL = 'agricultural.io.vn:9000'
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET = 'minio123'
PATH = "./file/"

def get_minio_object(filename):
    print(f'get data {filename} in minio')
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)
    client.fget_object(BUCKET_NAME, filename, PATH + filename)
    return PATH + filename