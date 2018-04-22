import pandas as pd
import numpy as np
import random as rnd
import seaborn
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline
import re

from scipy.stats import boxcox
from functools import reduce

def replaceWithBC(data, attr):
    bc, att = boxcox(data[attr].dropna())
    i2 = 0
    for i in range(data.shape[0]):
        if pd.notna(data.loc[i, attr]):
            data.loc[i, attr] = bc[i2]
            i2 = i2 + 1
    data.rename(index=str, columns={attr: attr + "_(box_cox)"})
            
def replaceOutliers(data, attr, thrshl, new):
    for i in range(data.shape[0]):
        if thrshl(data.loc[i, attr]):
            data.loc[i, attr] = new
            
def replaceOutliersD(data, attr, thrshl, new):
    for i in range(data.shape[0]):
        if thrshl(data.loc[i, attr]):
            data.loc[i, attr] = new(data.loc[i])


def preprocess(filepath):
    #1 nacitanie
    print('1 nacitanie')
    train_data = pd.DataFrame()
    train_data = pd.read_csv(filepath)

    #2 nahradenie indexu
    print('2 nahradenie indexu')
    train_data = train_data.set_index('Unnamed: 0')
    train_data.index.names = ['id']

    #3 rozvinutie stplcov
    print('3 rozvinutie stplcov')
    import json
    raw = train_data.loc[1, 'medical_info']
    raw = raw.replace("'", '"')
    data = json.loads(raw)

    #pridanie stlpcov
    for k in data.keys():
        train_data[k] = np.nan

    #naplnenie zaznamov hodnotami z medical_info
    for i in range(train_data.shape[0]):
        raw = train_data['medical_info'][i]
        raw = raw.replace("'", '"')
        data = json.loads(raw)
        for att in data.keys():
            train_data.loc[i, att] = data[att]
        
    #odstranenie nepotrebneho stlpca
    train_data = train_data.drop(columns=['medical_info'])

    #4 ciselne FTI
    print('4 ciselne FTI')
    train_data['FTI'] = pd.to_numeric(train_data['FTI'], errors='coerce')

    #5 ciselne TBG
    print('5 ciselne TBG')
    train_data['TBG'] = pd.to_numeric(train_data['TBG'], errors='coerce')
    
    train_data['TT4'] = pd.to_numeric(train_data['TT4'], errors='coerce')

    #6 rozvinutie vysledku testu
    print('6 rozvinutie vysledku testu')
    for i in range(train_data.shape[0]):
        train_data.loc[i, 'test'], train_data.loc[i, 'testID'] = str(train_data['class'][i]).split(".|")

    train_data = train_data.drop(columns=['class'])

    #7 zahodenie nepotrebnych stlpcov
    print('7 zahodenie nepotrebnych stlpcov')
    train_data = train_data.drop(['TBG measured', 'TBG', 'education', 'testID'], axis=1)

    #8 rozvinutie datumu narodenia
    print('8 rozvinutie datumu narodenia')
    train_data['birth_year'] = np.nan
    train_data['birth_month'] = np.nan
    
    for i in range(train_data.shape[0]-1):
        if (pd.isna(train_data.loc[i]['date_of_birth'])):
            continue
        date = list(map(int, re.split(' |-|/|:', train_data.iloc[i]['date_of_birth'])))
        if date[0] <= 31 and date[2] <= 31:
            continue
        elif date[0] <= 31 and date[2] > 31:
            b = date[2]
            date[2] = date[0]
            date[0] = b
        if date[0] < 1900:
            if int(str(date[0])[-2:]) < 18:
                date[0] = date[0] + 2000 - int(str(date[0])[:-2] if str(date[0])[:-2] != '' else 0) * 100
            else:
                date[0] = date[0] + 1900 - int(str(date[0])[:-2] if str(date[0])[:-2] != '' else 0) * 100
        if len(date) > 3:
            date = date[:3]
        train_data.loc[i, 'birth_year'] = date[0]
        train_data.loc[i, 'birth_month'] = date[1]

    train_data['birth_month'] = train_data['birth_month'].astype(str)
    for i in range(train_data.shape[0]):
        if train_data.loc[i,'birth_month'] == 'nan':
            train_data.loc[i,'birth_month'] = '?'
            
    #9 zlucenie hodnot
    print('9 zlucenie hodnot')
    for i in range(train_data.shape[0]):
        if train_data.loc[i, 'on thyroxine'] == 'FALSE' or train_data.loc[i, 'on thyroxine'] == 'F':
            train_data.loc[i, 'on thyroxine'] = 'f'
        elif train_data.loc[i, 'on thyroxine'] == 'TRUE' or train_data.loc[i, 'on thyroxine'] == 'T':
            train_data.loc[i, 'on thyroxine'] = 't'
        
        for col in train_data.columns[train_data.dtypes == np.object]:
            if pd.notna(train_data.loc[i, col]):
                train_data.loc[i, col] = train_data.loc[i, col].lower().strip().replace("-", "_")
        
    #10 nahrada za box-cox
    print('10 nahrada za box-cox')
    replaceWithBC(train_data, 'TSH')
    replaceWithBC(train_data, 'fnlwgt')

    #11 odstranenie odlahlych hodnot
    print('11 odstranenie odlahlych hodnot')
    replaceOutliers(train_data, 'fnlwgt', lambda a: a > 720, np.percentile(train_data['fnlwgt'], 95))
    replaceOutliers(train_data, 'hours-per-week', lambda a: a > 95, np.percentile(train_data['hours-per-week'], 95))
    replaceOutliersD(train_data, 'age', lambda a: a > 100, lambda a: 2017 - a.birth_year)
    
    #12 zlucenie education-num
    print('12 zlucenie education-num')
    for i in range(len(train_data['education-num'])):
        if train_data.loc[i, 'education-num'] < 0:
            train_data.loc[i, 'education-num'] *= -1
        if train_data.loc[i, 'education-num'] >= 100:
            train_data.loc[i, 'education-num'] = train_data.loc[i, 'education-num']/100
        
    #13 kategorizacia kapitalu
    print('13 kategorizacia kapitalu')
    train_data['yield'] = np.nan
    for i in range(train_data.shape[0]):
        yld = train_data.loc[i,'capital-gain'] - train_data.loc[i, 'capital-loss']
        if (yld == 0):
            train_data.loc[i, 'yield'] = 'none'
        elif (yld < 0):
            train_data.loc[i, 'yield'] = 'loss'
        else:
            train_data.loc[i, 'yield'] = 'gain'

    train_data = train_data.drop(['capital-gain', 'capital-loss'], axis=1)

    #14 nahradzovanie priemerom
    print('14 nahradzovanie priemerom')
    import math

    mean_med = train_data.copy()

    for column in mean_med.columns:
        if np.issubdtype(mean_med[column].dtype, np.number):
            if mean_med[column].std() / mean_med[column].mean() <= 1:
                mean = mean_med[column].mean()
                for i in range(len(mean_med[column])):
                    if (np.isnan(mean_med.loc[i, column])):
                        mean_med.loc[i, column] = mean
            else:
                median = mean_med[column].median()
                for i in range(len(mean_med[column])):
                    if (np.isnan(mean_med.loc[i, column])):
                        mean_med.loc[i, column] = median

    for i in range(mean_med.shape[0]):
        mean_med.loc[i, 'birth_year'] = math.ceil(mean_med.loc[i, 'birth_year'])
       
    #15 nahradzovanie regresiou
    print('15 nahradzovanie regresiou')
    import sklearn.linear_model
    import sklearn.model_selection

    linreg = train_data.copy()
    numericOnly = pd.DataFrame()
    scores = {}

    for col in train_data.columns:
        if np.issubdtype(train_data[col].dtype, np.number):
            numericOnly[col] = train_data[col].dropna()

    for col in train_data.columns:
        if np.issubdtype(train_data[col].dtype, np.number):
            numericOnly = numericOnly.dropna()
            X = numericOnly
            y = numericOnly[col]
            X = X.drop(col, axis=1)
            
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            
            reg = sklearn.linear_model.LinearRegression()
            reg.fit(X_train, y_train)
            print("Presnosť pre", col, ":", reg.score(X_test, y_test))
            
            scores[col] = reg.score(X_test, y_test)
            
            notknown = pd.DataFrame()
            for col2 in train_data.columns:
                if np.issubdtype(train_data[col2].dtype, np.number):
                    #kvôli neznámym hodnotam v necieľových stĺpcoch musíme použiť odhad neznámych dohnôt priemerom/mediánom
                    notknown[col2] = mean_med[col2]
                if col2 == col:
                    #iba pre cieľový stĺpec vyberáme dáta z pôvodného súboru, kde sú stále aj neznáme hodnoty
                    notknown[col2] = train_data[col2]
            
            notknown = notknown[pd.isna(notknown[col])]
            if (notknown.shape[0] == 0):
                continue
                
            X = notknown
            y = notknown[col]
            X = X.drop(col, axis=1)
                    
            y = reg.predict(X)
            
            #postupne dopĺňame neznáme hodnoty hodnotami odhadovanými lineárnou regresiou
            i2 = 0
            for i in range(linreg.shape[0]):
                if (pd.isna(linreg.loc[i, col])):
                    linreg.loc[i, col] = y[i2]
                    i2 = i2 + 1

    for i in range(mean_med.shape[0]):
        linreg.loc[i, 'birth_year'] = math.ceil(linreg.loc[i, 'birth_year'])

    #16 nahradenie priemerom/medianom a regresiou
    print('16 nahradenie priemerom/medianom a regresiou')
    raw_data = train_data.copy()

    for col in train_data.columns:
        if np.issubdtype(train_data[col].dtype, np.number):
            if scores[col] > 0.75:
                train_data[col] = linreg[col]
            else:
                train_data[col] = mean_med[col]

    #17 nahradenie ? za NaN
    print('17 nahradenie ? za NaN')
    for col in train_data.columns[train_data.dtypes == np.object]:
        for i in range(train_data.shape[0]):
            if train_data.loc[i, col] == '?':
                train_data.loc[i, col] = np.nan

    import random
    
    #18 nahradzovanie KNN
    print('18 nahradzovanie KNN')
    from sklearn import neighbors
    from sklearn.preprocessing import StandardScaler

    knr = train_data.copy()

    i = 0
    while knr[knr.columns[knr.dtypes == np.object]].isnull().values.any() and i < 49:
        allNumeric = pd.get_dummies(knr, columns=knr.columns[knr.dtypes == np.object], dummy_na=True)

        NanColumns = []
        for col in allNumeric.columns[11:]:
            if '_nan' in col:
                NanColumns.append(col)

        for col in train_data.columns[train_data.dtypes == np.object]:

            X = allNumeric.copy()
            X = X.drop(NanColumns, axis=1)
            
            valcols = []
            nancol = ''
            for col2 in allNumeric.columns[11:]:
                if col2[:len(col)] == col:
                    if '_nan' not in col2:
                        valcols.append(col2)
                    else:
                        nancol = col2

            if (allNumeric[nancol] == 0).all():
                continue

            print(col, "Ostava NaN: ", allNumeric[allNumeric[nancol] == 1].shape[0])

            for currcol in valcols:
                y = allNumeric[currcol].copy()
                
                scaler = StandardScaler()
                scaler.fit(X)
                X = pd.DataFrame(scaler.transform(X), columns=X.columns)
                
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
                clf = neighbors.KNeighborsClassifier(50 - i, weights='uniform')
                clf.fit(X_train, y_train)

                Q = allNumeric[allNumeric[nancol] == 1].drop(NanColumns, axis=1)
                Q = pd.DataFrame(scaler.transform(Q), columns=Q.columns)
                
                y = clf.predict(Q)
                
                #postupne dopĺňame neznáme hodnoty hodnotami odhadovanými podla najblizsich susedov
                i2 = 0
                for i3 in range(knr.shape[0]):
                    if (pd.isna(knr.loc[i3, col])):
                        if y[i2] != 0:
                            knr.loc[i3, col] = currcol[len(col)+1:]
                        i2 = i2 + 1
                        
        i = i + 1

    #19 nahradzovanie KNN
    print('19 hotovo')
    train_data = knr

    return train_data
