import os
import os.path
import shutil
import fileinput

config_filepath = "dag_config/"
dag_template_filename = "dag_template.py"


def create_dags_file(dag_name, data_url, model, argument, agricultural_name, user_name, type=None):
    try:
        dirname = os.path.dirname(__file__)
        new_filename = "dags/" + dag_name + ".py"
        print(dag_name)
        print(new_filename)
        template_file = os.path.join(dirname, dag_template_filename)
        dag_dir = dirname.replace("include", "")
        dag_file = os.path.join(dag_dir, new_filename)
        shutil.copyfile(template_file, dag_file)
        file_input = fileinput.input(dag_file, inplace=True)
        for line in file_input:
            line = line.replace("{dag_id_to_replace}", dag_name)
            line = line.replace("{model_id}", dag_name)
            line = line.replace("{model_name}", model)
            line = line.replace("{data_name}", data_url)
            if type:
                line = line.replace("{type}", type)
            line = line.replace("{argument}", str(argument))
            if agricultural_name:
                line = line.replace("{agricultural_name}", agricultural_name)
            if user_name:
                line = line.replace("{user_name}", user_name.strip())
            print(line, end="")
        print("DAG " + dag_file + " created")
        file_input.close()
        return True
    except Exception as e:
        print("DAG " + dag_name + " failed " + str(e))
        file_input.close()
        return False


def check_dag_exists(dag_name):
    dirname = os.path.dirname(__file__)
    dag_dir = dirname.replace("include", "")
    filename = "dags/" + dag_name + ".py"
    dag_file = os.path.join(dag_dir, filename)
    return os.path.isfile(dag_file)


if __name__ == "__main__":
    create_dags_file("test_LSTM",
                     "https://raw.githubusercontent.com/thuongh2/FinalProject/main/data/data_price.csv",
                     "LSTM",
                     {
                         "size": 0.8,
                         "timestep": 10,
                         "epochs": 5,
                         "batchsize": 30,
                         "layers_data": [
                             {
                                 "id": "LAYER 1",
                                 "units": "10"
                             }
                         ]
                     }, "LUA", "thuong", "DIFF")
