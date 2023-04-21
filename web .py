from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])


def predict():
    Sepal_Length=request.form.get("sl")
    Sepal_Width=request.form.get("sw")
    Petal_Length=request.form.get("pl")
    Petal_Width=request.form.get("pw")

    result=model.predict(np.array([Sepal_Length,Sepal_Width,Petal_Length,Petal_Width]).reshape(1,4))
    if result[0]=='Iris-setosa':
       result="Predicted flower is Iris-setosa"
    elif result[0]=='Iris-versicolor':
       result="Predicted flower is Iris-versicolor"
    elif result[0]=='Iris-virginica':
       result="Predicted flower is Iris-virginica"
    else:
       result="Others"
    
    return render_template('res.html',result=result)

if __name__=='__main__':
    app.run(port=8000)
    