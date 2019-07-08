import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
import pandas as pd
import pickle


UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL = pickle.load(open('./models/Random_Forest_model.pkl', 'rb'))

def predict(file_path):
    X = pd.read_csv(file_path, delimiter = ',')
    X = X[['athlete','financialAid','gender','geography','highschool',\
                      'legacy','major','orientation','race','year','school']]
    X = X.fillna(X.mean())
    return MODEL.predict(X)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            res = predict(file_path)
            res = [str(int(x)) for x in list(res)]
            return '<br>'.join(res)

    return """
    <!doctype html>
    <title>2020 Presidential Election Prediction</title>
    <h1>2020 Presidential Election Prediction</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
