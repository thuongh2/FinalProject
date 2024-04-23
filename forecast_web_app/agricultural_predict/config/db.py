from flask import Flask
from flask_pymongo import pymongo

DB_URI = "mongodb+srv://finalproject:j7TKpBLGsfhg6MF6@cluster0.ipzpp93.mongodb.net/agricultural_predict?retryWrites=true&w=majority"

client = pymongo.MongoClient(DB_URI)

db = client.get_database('agricultural_predict')

model = pymongo.collection.Collection(db, 'model')

user = pymongo.collection.Collection(db, 'user')

train_model = pymongo.collection.Collection(db, 'train_model')