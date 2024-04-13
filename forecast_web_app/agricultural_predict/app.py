from flask import Flask
from flask import render_template
from flask import send_from_directory

from pymongo import MongoClient

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/admin')
def admin():
    return render_template('admin/index.html')

@app.route('/hello/')
@app.route('/hello/<name>')
def hello_name(name=None):
    return render_template('hello.html', name=name)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)