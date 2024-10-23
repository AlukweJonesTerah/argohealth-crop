from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, \
    TextAreaField, FloatField
from wtforms.validators import DataRequired, Length, Email, EqualTo, InputRequired, NumberRange
from wtforms import ValidationError
from home.models import User
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask import flash


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField(
        'Password', validators=[DataRequired(), Length(min=8, max=30)]
    )
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Log In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField(
        'Password', validators=[DataRequired(), Length(min=8, max=30)])
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[
            DataRequired(), Length(min=8, max=30), EqualTo('password')
        ])
    submit = SubmitField('Submit')


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    about_me = StringField('About me', validators=[Length(min=0, max=140)])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')


class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileRequired(), FileAllowed(['jpg', 'png'], 'Images only!')],
                      render_kw={"class": "form-control"})
    submit = SubmitField('Upload', render_kw={"class": "btn btn-primary"})

class CropRecommendationForm(FlaskForm):
    nitrogen = FloatField('Nitrogen (kg/ha)', validators=[InputRequired(), NumberRange(min=0, max=1000)],
                          render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                     "placeholder": "Ratio of Nitrogen content in soil - kg/ha"})
    phosphorous = FloatField('Phosphorous (kg/ha)', validators=[InputRequired(), NumberRange(min=0, max=1000)],
                             render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                        "placeholder": "Ratio of Phosphorous content in soil - kg/ha"})
    potassium = FloatField('Potassium (kg/ha)', validators=[InputRequired(), NumberRange(min=0, max=1000)],
                           render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                      "placeholder": "Ratio of Potassium content in soil - kg/ha"})
    temperature = FloatField('Temperature (C)', validators=[InputRequired(), NumberRange(min=-50, max=50)],
                             render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                        "placeholder": "Temperature in degree Celsius"})
    humidity = FloatField('Humidity', validators=[InputRequired(), NumberRange(min=0, max=100)],
                          render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                     "placeholder": "Relative humidity in %"})
    ph = FloatField('pH of soil', validators=[InputRequired(), NumberRange(min=0, max=14)],
                    render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                               "placeholder": "pH value of the soil"})
    rainfall = FloatField('Rainfall (mm)', validators=[InputRequired(), NumberRange(min=0, max=1000)],
                          render_kw={"class": "form-control", "step": "0.0001", "type": "number",
                                     "placeholder": "Rainfall in mm"})
    submit = SubmitField('Submit', render_kw={"class": "btn btn-primary"})

    def validate(self, extra_validators=None):
        if not super().validate():
            return False

        # Custom validation logic
        # Ensure that rainfall is less than humidity
        if self.rainfall.data > self.humidity.data:
            flash('Rainfall should be less than humidity.', 'error')
            return False

        return True
