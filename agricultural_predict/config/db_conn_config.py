from flask_pymongo import pymongo
import base64

DB_COON = "bW9uZ29kYitzcnY6Ly9maW5hbHByb2plY3Q6ajdUS3BCTEdzZmhnNk1GNkBjbHVzdGVyMC5pcHpwcDkzLm1vbmdvZGIubmV0L2FncmljdWx0dXJhbF9wcmVkaWN0P3JldHJ5V3JpdGVzPXRydWUmdz1tYWpvcml0eQ=="

# Decode the Base64 encoded connection string
DB_URI = base64.b64decode(DB_COON).decode("utf-8")

# Initialize MongoDB client
client = pymongo.MongoClient(DB_URI)

# Get the database
db = client.get_database('agricultural_predict')

# Collections
model_info_collection = db['model_info']
user_collection = db['user']
model_registry_collection = db['model_registry']
model_data_relations_collection = db['model_data_relations']

# Optionally, add print statements to verify the connection and collections
print(f"Connected to database: {db.name}")
print("Collections in the database:")
print(db.list_collection_names())