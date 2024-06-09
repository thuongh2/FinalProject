from flask_pymongo import pymongo
import base64

DB_COON = "bW9uZ29kYitzcnY6Ly9maW5hbHByb2plY3Q6ajdUS3BCTEdzZmhnNk1GNkBjbHVzdGVyMC5pcHpwcDkzLm1vbmdvZGIubmV0L2FncmljdWx0dXJhbF9wcmVkaWN0P3JldHJ5V3JpdGVzPXRydWUmdz1tYWpvcml0eQ=="

DB_URI = base64.b64decode(DB_COON)

client = pymongo.MongoClient(str(DB_URI.decode("utf-8")))

db = client.get_database('agricultural_predict')

model = pymongo.collection.Collection(db, 'model')

user = pymongo.collection.Collection(db, 'user')

train_model = pymongo.collection.Collection(db, 'train_model')

model_relationship_collection = pymongo.collection.Collection(db, 'model_relationship')