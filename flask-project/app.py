from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/tryModel')
def subpage1():
    return render_template('tryModel.html')

@app.route('/about')
def subpage2():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)

