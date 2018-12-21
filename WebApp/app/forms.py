from flask_wtf import Form
from wtforms import StringField, BooleanField, TextAreaField, HiddenField, SubmitField
from wtforms.validators import DataRequired


class ThatsWhatSheSaidForm(Form):
	potentialTWSS = StringField('thatswhatshesaid')
	

class ValidationButton(Form):
	hiddenText = HiddenField('text', validators=[DataRequired()])
	hiddenTWSSresult = HiddenField('twssResult', validators=[DataRequired()])
#	isTwss = BooleanField('isTwss', validators=[DataRequired()])
	isTwss = SubmitField('isTwss')
	isNotTwss = SubmitField('isNotTwss')

