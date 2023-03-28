import joblib
from flask import Flask, render_template, session, url_for, redirect
from tensorflow.keras.models import load_model

from Run_Predictions import return_prediction
from Web_Form import InputForm
from config import MODEL_PATH, SCALER_PATH, API_KEY

# load the saved model
loaded_iris_model = load_model(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)
app.config['SECRET_KEY'] = API_KEY

# optional home route
@app.route('/', methods=['GET', 'POST'])

def index():
    # create instance of form
    form = InputForm()
    # if filled submit data to session
    if form.validate_on_submit():
        session['s_len'] = form.s_len.data
        session['s_w'] = form.s_w.data
        session['p_len'] = form.p_len.data
        session['p_w'] = form.p_w.data
        # forward data to API
        return redirect(url_for("iris_class_prediction"))

    return render_template('home.html', form=form)

# expect JSON POST to forward to prediction model
@app.route('/api/iris')

def iris_class_prediction():
    content = {}
    content['s_len'] = float(session['s_len'])
    content['s_w'] = float(session['s_w'])
    content['p_len'] = float(session['p_len'])
    content['p_w'] = float(session['p_w'])

    results = return_prediction(loaded_iris_model, loaded_scaler, content)
    return render_template('prediction.html', results=results)


if __name__ == '__main__':
    app.run()