# Setting Airflow
# Ref: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

fetching_file:
	echo "Fetching docker-compose.yaml file"
	curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.1/docker-compose.yaml'

setting_env:
	mkdir -p ./dags ./logs ./plugins ./config
	echo -e "AIRFLOW_UID=$(id -u)" > .env

init:
	echo "Initializing Airflow"
	docker-compose up airflow-init
	echo "Starting Airflow"
	docker-compose up -d