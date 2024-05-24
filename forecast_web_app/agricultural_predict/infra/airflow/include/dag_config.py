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
        print(new_filename)
        template_file = os.path.join(dirname, dag_template_filename)
        dag_dir = dirname.replace("include", "")
        dag_file = os.path.join(dag_dir, new_filename)
        shutil.copyfile(template_file, dag_file)

        for line in fileinput.input(dag_file, inplace=True):
            line = line.replace("{dag_id_to_replace}", dag_name)
            line = line.replace("{model_name}", model)
            line = line.replace("{data_name}", data_url)
            line = line.replace("{type}", type)
            line = line.replace("{argument}", str(argument))
            line = line.replace("{agricultural_name}", agricultural_name)
            line = line.replace("{user_name}", user_name)
            print(line, end="")
        print("DAG " + dag_file + " created")
        return True
    except:
        print("DAG " + dag_name + " failed")
        return False


def check_dag_exists(dag_name):
    dirname = os.path.dirname(__file__)
    dag_dir = dirname.replace("include", "")
    filename = "dags/" + dag_name + ".py"
    dag_file = os.path.join(dag_dir, filename)
    return os.path.isfile(dag_file)


if __name__ == "__main__":
    create_dags_file("test_12", "https://raw.githubusercontent.com/thuongh2/FinalProject/main/data/var_data.csv",
                     "VAR", {'P': 10, 'size': 0.8}, "LUA", "test","DIFF");
