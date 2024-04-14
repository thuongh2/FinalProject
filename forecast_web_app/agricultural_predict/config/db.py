from flask import Flask
from flask_pymongo import pymongo

DB_URI = "mongodb+srv://admin:cqH3HoQ1Cp4zafXn@cluster0.putsxyw.mongodb.net/agricultural_predict?retryWrites=true&w=majority"

client = pymongo.MongoClient(DB_URI)

db = client.get_database('agricultural_predict')

model = pymongo.collection.Collection(db, 'model')