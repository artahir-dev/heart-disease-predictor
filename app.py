##############################################################
# LIBRARIES

from flask import Flask, render_template, request, redirect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from math import isnan
import pandas as pd
import os


##############################################################
# INITIALIZATIONS

app = Flask(__name__)
app.config['DF'] = pd.read_csv('data/heart.csv')
app.config['MODEL'] = None
app.config['REMOVE_NULL_VALUES'] = 0
app.config['LABEL_ENCODING'] = 0
app.config['TRAIN_MODEL'] = 0

app.config['LABELS'] = {
    'Gender': {},
    'ChestPainType': {},
    'RestingECG': {},
    'ExerciseAngina': {},
    'ST_Slope': {}
}

##############################################################
# ROUTES

@app.route('/')
def f0():
    return render_template('home.html')

@app.route('/admin-pannel')
def f1():
    return render_template('template.html')

#################
@app.route('/show-dataset')
def f2():

    # if the request of row count is not valid then return template.html
    if (request.args.get('row').isdigit() == False):
        return render_template('template.html')
    
    # fetching row count
    row_count = int(request.args.get('row'))

    # fetching data from csv file
    column_names = app.config['DF'].columns.tolist()

    if (row_count > 0):
        r = app.config['DF'].head(row_count)
        Data = r.values.tolist()
        removeFloatValues(Data)
    else:
        Data = []
    
    Data.insert(0, column_names)

    # rendering show_dataset.html with Data
    return render_template('show_dataset.html', data=Data)

#################
@app.route('/remove-null-values')
def f3():
    
    if (app.config['REMOVE_NULL_VALUES'] == 1 ):
        return render_template('message.html', message="Already removed null values")

    # removing null values from age column
    for i in range(2):
        x = app.config['DF'].loc[app.config['DF']['HeartDisease'] == i, 'Age']
        app.config['DF'].loc[app.config['DF']['HeartDisease'] == i, 'Age'] = x.fillna(x.mean())

    app.config['DF']['Age'] = app.config['DF']['Age'].astype(int)

    # removing null values from gender column
    for i in range(2):
        x = app.config['DF'].loc[app.config['DF']['HeartDisease'] == i, 'Gender']
        app.config['DF'].loc[app.config['DF']['HeartDisease'] == i, 'Gender'] = x.fillna(x.mode()[0])

    app.config['REMOVE_NULL_VALUES'] = 1
    
    return redirect('/show-dataset?row=10')

#################
@app.route('/label-encoding')
def f4():

    if (app.config['REMOVE_NULL_VALUES'] == 0 ):
        return render_template('message.html', message="First remove null values")
    
    if (app.config['LABEL_ENCODING'] == 1 ):
        return render_template('message.html', message="Already encoded the labels")

    encoder = LabelEncoder()

    for col in app.config['LABELS'].keys():

        categories = app.config['DF'][col].tolist()
        labels = encoder.fit_transform(categories).tolist()
        
        app.config['DF'][col] = labels

        dict = {label: category for label, category in zip(labels, categories)}
        app.config['LABELS'][col] = dict

    app.config['LABEL_ENCODING'] = 1
    return redirect('/show-dataset?row=10')
 
#################
@app.route('/data-visualization')
def f5():

    if (app.config['REMOVE_NULL_VALUES'] == 0 ):
        return render_template('message.html', message="First remove null values")

    if (not graphsCreated()):
        create_graph1()
        create_graph2()
        create_graph3()
        create_graph4()

    return render_template('/graphs/visualize.html')
 
#################
@app.route('/train-model')
def f6():
    
    if app.config['TRAIN_MODEL'] == 0:

        if (app.config['REMOVE_NULL_VALUES'] == 0 ):
            return render_template('message.html', message="First remove null values")
        
        if (app.config['LABEL_ENCODING'] == 0 ):
            return render_template('message.html', message="First encode the labels")
        
        X = app.config['DF'].iloc[:, 0:11]
        y = app.config['DF']['HeartDisease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)

        app.config['MODEL'] = RandomForestClassifier(n_estimators=160, random_state=40)

        app.config['MODEL'].fit(X_train, y_train)

        app.config['TRAIN_MODEL'] = 1
    
    else:
        return render_template('message.html', message="Model already trained")


    return render_template('message.html', message="Model trained successfully")

#################
@app.route('/user-pannel')
def f7():
    return render_template('user_pannel.html', variant=0)

#################
@app.route('/predict', methods=['POST'])
def f8():
    if (app.config['TRAIN_MODEL'] == 0):
        return render_template('user_pannel.html', variant=1)

    features = app.config['DF'].columns.tolist()[:-1]
    x = [request.form.get(feature) for feature in features]

    prepareX(x)

    prediction = app.config['MODEL'].predict([x])[0]

    if(prediction == 0):
        return render_template('user_pannel.html', variant=2)

    else:
        return render_template('user_pannel.html', variant=3)

#################
@app.route('/reset')
def f10():
    app.config['DF'] = pd.read_csv('data/heart.csv')
    app.config['REMOVE_NULL_VALUES'] = 0
    app.config['LABEL_ENCODING'] = 0
    app.config['TRAIN_MODEL'] = 0

    return redirect('/show-dataset?row=10')

#################

##############################################################
# FUNCTIONS

def removeFloatValues(Data):
    for i in range(len(Data)):
        for j in range(len(Data[i])):
            if ( j!= 9 and not type(Data[i][j]) is str and not isnan(Data[i][j])):
                Data[i][j] = int(Data[i][j])

#################
def create_graph1():
    M = 'M'
    F = 'F'
    if (app.config['LABEL_ENCODING'] == 1 ):
        for key, value in app.config['LABELS']['Gender'].items():
            if value == 'M':
                M = key
            if value == 'F':
                F = key

    m1 = len(app.config['DF'][(app.config['DF']['Gender']==M) & (app.config['DF']['HeartDisease']==1)])
    m2 = len(app.config['DF'][(app.config['DF']['Gender']==M) & (app.config['DF']['HeartDisease']==0)])
    f1 = len(app.config['DF'][(app.config['DF']['Gender']==F) & (app.config['DF']['HeartDisease']==1)])
    f2 = len(app.config['DF'][(app.config['DF']['Gender']==F) & (app.config['DF']['HeartDisease']==0)])

    fig1 = go.Figure()

    x = ['Male', 'Female']
    y1 = [m1, f1]
    y2 = [m2, f2]

    fig1.add_trace(go.Bar(x=x, y=y1, name='have heart disease', marker={'color': 'red'}, width=0.4))
    fig1.add_trace(go.Bar(x=x, y=y2, name='have no heart disease', marker={'color': 'blue'}, width=0.4))

    fig1.update_layout(barmode='stack', template='plotly_dark', yaxis_title='Count', width=800, height=550)
    fig1.write_html("templates/graphs/g1.html")

#################
def create_graph2():
    ages = app.config['DF']['Age'].unique()
    y1 = []
    y2 = []

    for age in ages:
        y1.append(len( app.config['DF'][ (app.config['DF']['Age']==age) & (app.config['DF']['HeartDisease']==1) ] ))
        y2.append(len( app.config['DF'][ (app.config['DF']['Age']==age) & (app.config['DF']['HeartDisease']==0) ] ))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=ages, y=y1, name='have heart disease', marker={'color': 'springgreen'}))
    fig2.add_trace(go.Bar(x=ages, y=y2, name='have no heart disease', marker={'color': 'yellow'}))

    fig2.update_layout(barmode='stack', template='plotly_dark', xaxis_title='Age', yaxis_title='count', width=800, height=550)

    fig2.write_html("templates/graphs/g2.html")

#################
def create_graph3():
    data = app.config['DF'][app.config['DF']['HeartDisease']==1]['RestingBP']

    fig3 = go.Figure()

    histogram_trace = go.Histogram(x=data, autobinx=False, nbinsx=23, marker=dict(color='deeppink'))

    fig3.add_trace(histogram_trace)

    fig3.update_layout(bargap=0.09, template='plotly_dark', xaxis_title='Blood pressure', yaxis_title='Heart  disease  cases')

    fig3.write_html("templates/graphs/g3.html")

#################
def create_graph4():
    data = app.config['DF'][app.config['DF']['HeartDisease']==1]['Cholesterol']

    fig4 = go.Figure()

    histogram_trace = go.Histogram(x=data, autobinx=False, nbinsx=30, marker=dict(color='gold'))

    fig4.add_trace(histogram_trace)

    fig4.update_layout(bargap=0.09, template='plotly_dark', xaxis_title='Cholesterol', yaxis_title='Heart  disease  cases')

    fig4.write_html("templates/graphs/g4.html")

#################
    
def graphsCreated():
    for i in range(1, 5):
        if(not os.path.exists(f'templates/graphs/g{i}.html')):
            return False
    return True
#################

def prepareX(x):
    features = app.config['DF'].columns.tolist()[:-1]
    
    for i in range(len(x)):
        if (features[i] in app.config['LABELS']):
            for key, value in app.config['LABELS'][features[i]].items():
                if (x[i] == value):
                    x[i] = key
        else:
            x[i] = float(x[i])
            if (i != 9):
                x[i] = int(x[i])

#################

##############################################################
# MAIN

if __name__ == "__main__":  
    app.run()

##############################################################

