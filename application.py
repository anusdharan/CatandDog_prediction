from flask import *
import sys
import os
from werkzeug.utils import secure_filename
from predict import predict_image
application = Flask(__name__)


@application.route('/')
def upload():
    return render_template("index.html")


@application.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'input_data', secure_filename(f.filename))
        f.save(file_path)




        outcome=predict_image(file_path)
        #print(outcome)
        #print(type(outcome))

        return str(outcome)
    return render_template("index.html")


if __name__ == '__main__':
    application.run(debug=True)