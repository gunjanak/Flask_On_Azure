from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', dynamic_text="This is the end or the beginning!")

if __name__ == '__main__':
    app.run()