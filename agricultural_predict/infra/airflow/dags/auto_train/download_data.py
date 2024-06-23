from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import time
import pandas as pd
import os
import shutil
from io import BytesIO
from minio import Minio
from airflow.decorators import dag, task, task_group
from datetime import datetime

# Set the original and new file paths
URL_DRIVE = "https://agro.gov.vn/vn/nguonwmy.aspx"
OS_FILE_PATH = "/opt/airflow/dags/celenium"
CAFE = "Ca-phe-voi-nhan-xo"
CAFE_OPTION_CHOISE = "Cà phê vối nhân xô"
LUA = "Gao-thuongRice"
LUA_OPTION_CHOISE = "Gạo thường|Rice"
XLS_EXTENSION = ".xls"
HTML_EXTENSION = ".html"

@dag(
    start_date = datetime(2024, 6, 8),
    schedule_interval='@daily',
    catchup=False,
    tags=['craw_data'],
)
def craw_data():

    @task()
    def get_data_from_web(visible_text):
        options = webdriver.ChromeOptions()
        prefs = {
            "profile.default_content_settings.popups": 0,
            "download.prompt_for_download": "false",
            "download.directory_upgrade": "true",
            "download.default_directory": "/tmp/downloads"
        }
        options.add_experimental_option("prefs", prefs)
        remote_webdriver = 'remote_chromedriver'
        with webdriver.Remote(f'{remote_webdriver}:4444/wd/hub', options=options) as driver:

            driver.get(URL_DRIVE)

            select_element = driver.find_element(By.ID, 'ctl00_maincontent_mathangnongsan')
            select = Select(select_element)
            if select:
                select.select_by_visible_text(visible_text)

                time.sleep(10)

                driver.execute_script("""
                    document.getElementById('ctl00_maincontent_tai_excel').removeAttribute("disabled");
                """)
                element = driver.find_element(By.ID, "ctl00_maincontent_tai_excel")
                element.click()

                time.sleep(30)
            else: 
                print("Fail when download file")


    @task()
    def conver_file_to_csv(file_type, fillter, file_name_upload):
        file_name = f'{OS_FILE_PATH}/{file_type}'
        os.rename(f'{file_name}{XLS_EXTENSION}', f'{file_name}{HTML_EXTENSION}')
        
        df_rice = pd.read_html(f'{file_name}{HTML_EXTENSION}')[0]
        
        try:
            # Đổi tên cột 'Ngày' thành 'date' và lọc lấy thị trường An Giang
            df_rice = df_rice.rename(columns={'Ngày': 'date'})
            df_rice['date'] = pd.to_datetime(df_rice['date'])
            df_rice.set_index('date', inplace=True)
            df_rice = df_rice[df_rice['Thị_trường'] == fillter]

                # Tính toán ngưỡng cho outlier sử dụng IQR
            Q1 = df_rice['Giá'].quantile(0.25)
            Q3 = df_rice['Giá'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Xác định các dòng dữ liệu có giá trị nằm ngoài ngưỡng
            outliers = df_rice[(df_rice['Giá'] < lower_bound) | (df_rice['Giá'] > upper_bound)]
            print("Số lượng outlier:", len(outliers))

            # Kiểm tra và thay đổi giá trị nếu nó nằm ngoài ngưỡng
            df_rice.at[df_rice.index[0], 'Giá'] = min(max(df_rice.iloc[0]['Giá'], lower_bound), upper_bound)

            # Thay thế giá trị của outlier bằng giá trị trước đó
            for index, row in outliers.iterrows():
                previous_date_index = df_rice.index.get_loc(index) - 1
                if previous_date_index >= 0:
                    previous_date = df_rice.index[previous_date_index]
                    previous_value = df_rice.loc[previous_date, 'Giá']
                    df_rice.at[index, 'Giá'] = previous_value
        except:
            print("Not handel outliner")
        df_rice = df_rice.rename(columns = {'Giá': 'price'})
        df_rice = df_rice.drop(['Thị_trường', 'Tên_mặt_hàng'], axis=1)

        fupload_object(df_rice, file_name_upload)

    def fupload_object(data, file_name_upload):
        BUCKET_NAME = 'data'
        MINIO_URL = 'agricultural.io.vn:9000'
        MINIO_ACCESS_KEY = 'minio'
        MINIO_SECRET = 'minio123'

        client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)

        csv_bytes = data.to_csv().encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

                # Make bucket if not exist.
        found = client.bucket_exists(BUCKET_NAME)
        if not found:
            client.make_bucket(BUCKET_NAME)
        else:
            print(f"Bucket {BUCKET_NAME} already exists")

        client.put_object(BUCKET_NAME,
                        file_name_upload,
                        data=csv_buffer,
                        length=len(csv_bytes),
                        content_type='application/csv')
        print('upload success')

    @task_group
    def get_data_lua():
        get_data_from_web(LUA_OPTION_CHOISE) >> conver_file_to_csv(LUA, 'An Giang', 'data_lua.csv')

    @task_group
    def get_data_cofe():
        get_data_from_web(CAFE_OPTION_CHOISE) >> conver_file_to_csv(CAFE, 'Đăk Lăk|Dak Lak', 'data_cafe.csv')
       
    @task(trigger_rule="all_done")
    def clear_file():
        folder = OS_FILE_PATH
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    [get_data_lua(), get_data_cofe()] >> clear_file()
    

craw_data = craw_data()