from flask import Flask, render_template
import io
from flask import Response
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


app = Flask(__name__)

url_datos = 'https://raw.githubusercontent.com/Rodrigo-JM108/Datos_IA/master/datos_ml.csv'
dataset = pd.read_csv(url_datos)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/prediccion", methods=['POST'] )
def prediccion():
    

    X = dataset[['duration', 'count']]
    Y = dataset['class']

    correos = dataset.groupby('class').size()

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

    modelo_RL = linear_model.LogisticRegression()

    modelo_RL.fit(X_train,y_train)


    y_pred = modelo_RL.predict(X_test)

    p_s = round(metrics.accuracy_score(y_test, y_pred),3)
    return render_template('index.html', prediccion=f'La predicción es: {p_s}', num_correos=f'{correos}')

@app.route('/grafica')
def dibuja_grafico():
    x_values = dataset['class'].unique()
    y_values = dataset['class'].value_counts().tolist()
    plt.bar(x_values, y_values)
    plt.title('Detección de SPAM')
    ax = plt.subplot()                   
    ax.set_xticks(x_values)            
    ax.set_xticklabels(x_values)        
    ax.set_xlabel('Tipo de correo')  
    ax.set_ylabel('Cantidad de correos') 
    output = io.BytesIO()
    plt.legend()
    plt.savefig(output, format='png') 
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run()