import datetime
import random
import base64
import string

def generate_id(prefix=None):
    """
    Tạo ra một ID theo định dạng "yymmdd + base64 random".
    
    Returns:
        str: ID được tạo.
    """
    # Lấy ngày tháng năm hiện tại
    now = datetime.datetime.now()
    
    # Tạo chuỗi "yymmdd"
    date_str = now.strftime("%y%m%d")
    
    # Tạo chuỗi "base64 random"
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    base64_random_str = base64.b64encode(random_str.encode()).decode()
    id_gen = date_str + base64_random_str
    if prefix:
        id_gen = prefix + id_gen
    
    return id_gen


def generate_string():
     # Tạo chuỗi "base64 random"
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    base64_random_str = base64.b64encode(random_str.encode()).decode()
    return base64_random_str