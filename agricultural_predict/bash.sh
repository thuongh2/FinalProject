#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the Flask application in debug mode
flask --app app.py run -p 5001 -h 0.0.0.0
