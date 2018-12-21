from flask import render_template, flash, redirect
from app import app, db, models
from .forms import ThatsWhatSheSaidForm, ValidationButton
from app.modules import ClassifierTrainer_TWSS
import datetime
from flask import request

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html',
                           title='Home')



@app.route('/whatshesaid', methods=['GET', 'POST'])
def whatshesaid():

  
  form = ThatsWhatSheSaidForm()
  validationButton = None

  if request.method == "POST":
    flash("POST request received.")

    if form.validate_on_submit():
      whatshesaidclassify(form)
      validationButton = ValidationButton(hiddenText = form.potentialTWSS.data)


  return render_template('thatswhatshesaid.html',
                         title='That\'s What She Said',
                         form=form,
                         validationButton = validationButton)



def whatshesaidclassify(form):
  inputtext = form.potentialTWSS.data
  result = ClassifierTrainer_TWSS.classify_text(inputtext)
  output = inputtext + ".... nah."
  
  if (result == 1):
    output = inputtext + "... THAT'S WHAT SHE SAID!"

  flash(output)
  return


@app.route("/whatshesaidvalidation", methods=['GET', 'POST'])
def whatshesaidvalidation():
  twss = models.TWSSsentence(sentence=validationButton.hiddenText, timestamp=datetime.datetime.now(), isTwss=validationButton.isTwss)
  db.session.add(twss)
  db.session.commit()
  flash("FISK")
  redirect('/whatshesaid')
















@app.route('/literarysummarizer', methods=['GET', 'POST'])
def literarysummarizer():
    return render_template('index.html',
                           title='Home')


@app.route('/lovecraftian', methods=['GET', 'POST'])
def lovecraftian():
    form = ThatsWhatSheSaidForm()
    if form.validate_on_submit():
        flash(form.thatswhatshesaid.data)
    return render_template('lovecraftian.html',
                           title='Home',
                           form=form)



