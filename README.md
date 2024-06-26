## HCMC UNIVERSITY OF TECHNOLOGY AND EDUCATION
### FACULTY OF INFORMATION TECHNOLOGY
### DEPARTMENT OF DATA ENGINEERING

## Project: APPLYING STATISTICAL METHODS AND MACHINE LEARNING TO PREDICT AGRICULTURAL PRICES

#### 1. PURPOSE OF THE PROJECT
The objective of this project is to develop an application to predict agricultural prices using statistical methods and machine learning. The main tasks include:
- Building and training models to predict agricultural prices using statistical time series analysis algorithms (ARIMA, ARIMAX, VAR, VARMA) and machine learning, deep learning methods (LSTM, GRU, BiLSTM).
- Developing a web application to predict agricultural prices with main features: displaying agricultural price predictions, allowing users to upload their own prediction models, training prediction models directly on the website, and enabling the system to automatically download new data and train models.

#### 2. Project Structure

- `agricultural_predict`: directory containing the web application
- `colab`: directory containing model training files
- `data_final`: directory containing agricultural data (csv)

#### 3. Application Deployment
**3.1 Deploy related services with Docker Compose**

```docker
airflow-webserver:
 image: hoaithuongdata/airflow-final-project
 ports:
 - "8080:8080"
Selenium:
 image: hoaithuongdata/selenium-final-project:v1.0
 ports:
 - "4444:4444"
minio:
 image: minio/minio
 ports:
 - "9000:9000"
 - "9001:9001"
mlflow:
 image: mlflow_server
 ports:
 - "5000:5000"
 ```

The application uses pre-provided images available on Docker Hub. Images such as airflow and selenium have been rebuilt with additional necessary libraries to meet the scope of the project.

3.2 Deploy MongoDB Atlas

Deploy MongoDB Atlas with a 512MB storage sandbox package and connect via pymongo. Create an account and set up MongoDB Atlas: Register or log in at MongoDB Atlas. Create a new project and cluster with the "M0 Sandbox" package (512MB storage). Set up the cluster and create a user. Create a user with access rights in "Database Access". Add IP Whitelist in "Network Access". Connect the web application via pymongo by getting the connection string from the cluster.
```
mongodb+srv://<username>:<password>@<clusteraddress>/<dbname>?retryWrites=true&w=majority
```

3.3 Deploy Web Server with Flask

Create and Activate Virtual Environment
Create a virtual environment:

```sh
python -m venv venv

source ./venv/Scripts/activate
```

Install the Packages for the application to function
```sh
pip install -r requirements.txt
```

Run the web application

Command to start the Flask interpreter running the application file. Run the application on port 5001 and -h 0.0.0.0 to allow access from any machine.

``` sh
flask --app app.py run -p 5001 -h 0.0.0.0
```

4. ACHIEVED RESULTS
- Understand and grasp the theory of statistical analysis and time series prediction.
- Understand and grasp the theory of deep learning, time series processing algorithms in deep learning.
- Understand and grasp methods and techniques to build agricultural price prediction models using statistical models and deep learning for time series analysis and prediction.
- Build univariate and multivariate agricultural price prediction models using statistical models (ARIMA, ARIMAX, VAR, VARMA) and deep learning (LSTM, GRU, BiLSTM).
- Develop a web application to display agricultural price predictions, allow users to upload their own prediction models, and train models directly on the website, with the system automatically downloading new data and training the models.