from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():

	if request.method == 'POST':
		a_inc = request.form['a']
		c_inc = request.form['b']
		ln_amt = request.form['c']
		term = request.form['d']
		cred = request.form['e']
		data = [a_inc,c_inc,ln_amt,term,cred]
		clean_data = [ float(i) for i in data]

		filename = "finalized_model.pkl"
		loaded_model = pickle.load(open(filename, 'rb'))
		ans=loaded_model.predict([clean_data])
		ans=int(ans)
	
	return render_template('predict.html' , prediction = ans)


if __name__ == '__main__':
	app.run(debug=True)
