#!/bin/bash

# Activate the virtual environment
source ./venv/Scripts/activate

# Install the required packages
pip install -r requirements.txt

# Run the Flask application in debug mode
flask --app app.py --debug run -p 5001
