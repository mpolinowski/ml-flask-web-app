from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class InputForm(FlaskForm):
    s_len = StringField("Sepal Length (cm)")
    s_w = StringField("Sepal Width (cm)")
    p_len = StringField("Petal Length (cm)")
    p_w = StringField("Petal Width (cm)")

    submit = SubmitField("Predict")