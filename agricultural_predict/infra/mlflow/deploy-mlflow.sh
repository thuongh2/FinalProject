#!/bin/bash

# Ref: https://github.com/sachua/mlflow-docker-compose
# Author: sachua, mathematicalmichael, ...

# Step 1: Clone the repository
echo "Cloning the repository..."
git clone https://github.com/sachua/mlflow-docker-compose.git

# Step 2: Change into the cloned directory
echo "Changing into the cloned directory..."
cd mlflow-docker-compose

# Step 3: Start the Docker containers
echo "Starting Docker containers..."
docker-compose up -d --build

echo "Done!"