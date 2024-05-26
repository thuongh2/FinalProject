# FinalProject

Create README.md 

How to run this

```shell
    cd infra     
```


Start airflow server
```shell
    cd airflow
    docker-compose up   
```
open in localhost:8080

username: airflow, password: airflow

Start MLFlow and Minio
```shell
    cd mlflow
    ./deploy-mlflow.sh 
 
```
open in host:5000 and host:9000

Minio username: minio, pass: minio123