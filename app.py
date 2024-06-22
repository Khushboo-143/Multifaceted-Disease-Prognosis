import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# def predict(values, dic):
    # diabetes
    # if len(values) == 10:
    #     dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
    #             'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
    #             'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}

    #     if dic['BMI'] <= 18.5:
    #         dic2['NewBMI_Underweight'] = 1
    #     elif 18.5 < dic['BMI'] <= 24.9:
    #         pass
    #     elif 24.9 < dic['BMI'] <= 29.9:
    #         dic2['NewBMI_Overweight'] = 1
    #     elif 29.9 < dic['BMI'] <= 34.9:
    #         dic2['NewBMI_Obesity 1'] = 1
    #     elif 34.9 < dic['BMI'] <= 39.9:
    #         dic2['NewBMI_Obesity 2'] = 1
    #     elif dic['BMI'] > 39.9:
    #         dic2['NewBMI_Obesity 3'] = 1

    #     if 16 <= dic['Insulin'] <= 166:
    #         dic2['NewInsulinScore_Normal'] = 1

    #     if dic['Glucose'] <= 70:
    #         dic2['NewGlucose_Low'] = 1
    #     elif 70 < dic['Glucose'] <= 99:
    #         dic2['NewGlucose_Normal'] = 1
    #     elif 99 < dic['Glucose'] <= 126:
    #         dic2['NewGlucose_Overweight'] = 1
    #     elif dic['Glucose'] > 126:
    #         dic2['NewGlucose_Secret'] = 1
        
    #     dic.update(dic2)
    #     values2 = list(map(float, list(dic.values())))
        
    #     model = pickle.load(open('models/diabetes_model2.sav','rb'))
        
    #     values = np.asarray(values2)
    #     return model.predict(values.reshape(1, -1))[0]
    
    # elif len(values) == 13:
    #     model = pickle.load(open('models/heart.pkl','rb'))
    #     values = np.asarray(values)
    #     return model.predict(values.reshape(1, -1))[0]
@app.route("/")
def home():
    return render_template('home.html')


@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')


@app.route("/parkinson", methods=['GET', 'POST'])
def parkinsonPage():
    return render_template('parkinson.html')

@app.route("/breast", methods=['GET', 'POST'])
def breastPage():
    return render_template('breast.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            
            for key, value in to_predict_dict.items():
                try:
                    
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)
                    
                
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            
            pred = predict(to_predict_list, to_predict_dict)
            
    except Exception as e:
        message = "Please enter valid data"
        print("Some error eccoured! ",e)
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

@app.route('/predictDiabetes',methods=['POST'])
def predDiab():
     
     p=request.form['Pregnancies']
     g=request.form['Glucose']
     b=request.form['BloodPressure']
     s=request.form['SkinThickness']
     i=request.form['Insulin']
     bm=request.form['BMI']
     dp=request.form['DiabetesPedigreeFunction']
     a=request.form['Age']
     l=[[p,g,b,s,i,bm,dp,a]]
     model = pickle.load(open('models/diabetes_model2.sav','rb'))
     print("result",model.predict(l))
     res=model.predict(l)[0]
     return render_template('predict.html', pred=res)

@app.route('/predictHeart',methods=['POST'])
def predHeart():
     
     a=request.form['age']
     s=request.form['sex']
     c=request.form['cp']
     trb=request.form['trestbps']
     ch=request.form['chol']
     fb=request.form['fbs']
     resec=request.form['restecg']
     th=request.form['thalach']
     ex=request.form['exang']
     olp=request.form['oldpeak']
     sl=request.form['slope']
     caa=request.form['ca']
     tha=request.form['thal']
     l1=[[a,s,c,trb,ch,fb,resec,th,ex,olp,sl,caa,tha]]
     model=pickle.load(open('models/heart_model.pkl','rb'))
     print("result",model.predict(l1))
     res2= model.predict(l1)[0]
     return render_template('predict.html',pred=res2)

@app.route('/predictParkinson',methods=['POST'])
def predparkin():
     
     a=request.form['MDVP:Fo(Hz)']
     b=request.form['MDVP:Fhi(Hz)']
     c=request.form['MDVP:Flo(Hz)']
     d=request.form['MDVP:Jitter(%)']
     e=request.form['MDVP:Jitter(Abs)']
     f=request.form['MDVP:RAP']
     g=request.form['MDVP:PPQ']
     h=request.form['Jitter:DDP']
     i=request.form['MDVP:Shimmer']
     j=request.form['MDVP:Shimmer(dB)']
     k=request.form['Shimmer:APQ3']
     l=request.form['Shimmer:APQ5']
     m=request.form['MDVP:APQ']
     n=request.form['Shimmer:DDA']
     o=request.form['NHR']
     p=request.form['HNR']
     q=request.form['RPDE']
     r=request.form['DFA']
     s=request.form['spread1']
     t=request.form['spread2']
     u=request.form['D2']
     v=request.form['PPE']
     
     l3=[[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v]]
     model=pickle.load(open('models/parkinson_model.pkl','rb'))
     print("result",model.predict(l3))
     res3= model.predict(l3)[0]
     return render_template('predict.html',pred=res3)

@app.route('/predictbreast',methods=['POST'])
def predbreast():
     
     a=request.form['id']
     b=request.form['radius_mean']
     c=request.form['texture_mean']
     d=request.form['perimeter_mean']
     e=request.form['area_mean']
     f=request.form['smoothness_mean']
     g=request.form['compactness_mean']
     h=request.form['concavity_mean']
     i=request.form['concave_points_mean']
     j=request.form['symmetry_mean']
     k=request.form['fractal_dimension_mean']
     l=request.form['radius_se']
     m=request.form['texture_se']
     n=request.form['perimeter_se']
     o=request.form['area_se']
     p=request.form['smoothness_se']
     q=request.form['compactness_se']
     r=request.form['concavity_se']
     s=request.form['concave_points_se']
     t=request.form['symmetry_se']
     tt=request.form['fractal_dimension_se']
     u=request.form['radius_worst']
     v=request.form['texture_worst']
     w=request.form['perimeter_worst']
     x=request.form['area_worst']
     y=request.form['smoothness_worst']
     z=request.form['compactness_worst']
     tha=request.form['concavity_worst']
     aa=request.form['concave_points_worst']
     bb=request.form['symmetry_worst']
     cc=request.form['fractal_dimension_worst']
     l1=[[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,tt,u,v,w,x,y,z,tha,aa,bb,cc]]
     model=pickle.load(open('models/breast_model.pkl','rb'))
     print("result",model.predict(l1))
     res2= model.predict(l1)[0]
     return render_template('predict.html',pred=res2)






if __name__ == '__main__':
    app.run(debug = True)

