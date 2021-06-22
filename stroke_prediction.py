# -*- coding: utf-8 -*-


import pandas as pd # bibliothèque pandas pour le prétraitement des données
df = pd.read_csv('healthcare-dataset-stroke-data.csv') # lire les données à l'aide de pandas
df.head()# montrant les 5 premières lignes .

"""### Les valeurs manquantes"""

# Calculez le nombre de valeurs manquantes pour chaque colonne
df.isnull().sum()

# le bmi est la colonne où se trouvent les valeurs manquantes. Ainsi, nous choisirons de remplacer ces valeurs NaN par la moyenne.
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

"""### Categorical features"""

"""
"other" ne représente pas grand-chose. Ainsi, 
et afin d'éviter les valeurs aberrantes, nous décidons de remplacer par le mode de la colonne genre.
"""
df['gender'] = df['gender'].replace('Other', list(df.gender.mode().values)[0])
df.gender.value_counts()# print the number of patients for each sex.

import matplotlib.pyplot as plt #Les bibliothèques Matplotlib et seaborn sont destinées aux visualisations.
import seaborn as sns
"""
Nous ferons les visualisations pour les fonctionnalités catégorielles.
"""
# La list des columns catégorielles
df_cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status', 'stroke']
fig, axs = plt.subplots(4, 2, figsize=(14,20))# L'intialization de la figure
axs = axs.flatten()
"""
parcourir chaque colonne de df_catd et le tracer
"""
for i, col_name in enumerate(df_cat):
    sns.countplot(x=col_name, data=df, ax=axs[i], hue =df['stroke']) # L'affichage de la visualization.
    plt.title("Bar chart of") # Titre
    axs[i].set_xlabel(f"{col_name}", weight = 'bold') # Libillé de l'axe x
    axs[i].set_ylabel('Count', weight='bold')# Libillé de l'axe Y

"""### Contunious features"""

"""
Visualisations pour les fonctionnalités continues
"""
df_num = ['age', 'avg_glucose_level', 'bmi'] # la list des fonctionnalités continues
fig, axs = plt.subplots(1, 3, figsize=(16,5)) # La figure
axs = axs.flatten()
"""
parcourir chaque colonne dans df_num et tracer
"""
for i, col_name in enumerate(df_num):
    sns.boxplot(x="stroke", y=col_name, data=df, ax=axs[i]) # La='affichage de la figure
    axs[i].set_xlabel("Stroke", weight = 'bold')# libellé de X
    axs[i].set_ylabel(f"{col_name}", weight='bold')# libellé de Y

"""### Features encoding"""

from sklearn.preprocessing import LabelEncoder # L'importation de l'objet LabelEncoder
"""
Encodage des caractéristiques catégorielles
"""
le = LabelEncoder() # La déclaration de l'objet
data = df # Gardez une copie des données avant l'encodage
data['gender'] = le.fit_transform(data['gender'])# L'encodage de gender
data['ever_married'] = le.fit_transform(data['ever_married'])# L'encodage de "ever_married"
data['work_type'] = le.fit_transform(data['work_type'])# L'encodage de "work_type"
data['Residence_type'] = le.fit_transform(data['Residence_type'])#L'encodage de "Residence_type"
data['smoking_status'] = le.fit_transform(data['smoking_status'])#L'encodage de "smoking_status"
df_en = data# df_en sont les nouvaux données aprés l'encodage.
df_en.head()#affichage de l'entêt

"""### Correlation between features"""

# nous utiliserons la carte thermique pour tracer la matrice de corrélation
corr = df_en.corr().round(2) # Le calcule de la matrice
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot = True); # L'affichage de la matrice avec heatmap

df_en.drop('ever_married',axis=1,inplace=True) # élimination de ever_maried.

"""### Data scaling"""

"""
Changement de L'échelle des données
"""
from sklearn.preprocessing import StandardScaler #importation de l'objet
s = StandardScaler()# Initialization de l'objet.
columns = ['avg_glucose_level','bmi','age']#La définition des columns à coder.
stand_scaled = s.fit_transform(df_en[['avg_glucose_level','bmi','age']])#l'encodage
stand_scaled = pd.DataFrame(stand_scaled,columns=columns)#La transformation des données codées à un 'dataframe"
df_en=df_en.drop(columns=columns,axis=1)
stand_scaled.head()#L'affichage de l'entêt des donnes codées.

df = pd.concat([df_en, stand_scaled], axis=1) # La concatenation des donnes codés avec les autres données
df.head(3)#l'affichade de l'entête

"""### Data split"""

# diviser les données en entries et en sorties.
X = df.drop(['id','stroke'],axis=1) #les entrées
y = df['stroke']# les sorties

# La division des données (Training / Test)
from sklearn.model_selection import train_test_split# importation de l'objet train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, accuracy_score, classification_report#l'importation des metriques d'evaluation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 124)# La dévision des données.

"""### Machine learning models"""

# Ici, nous importons 3 modèles d'apprentissage automatique
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Random Forest
model = RandomForestClassifier() # Initialization de modéle.
model.fit(x_train,y_train) # l'entrainment du modéle.
pred = model.predict(x_test) # La prédiction dans les données de Test
print('\n Les performances de \'Fôret aléatoire\''+10*'-')
print(confusion_matrix(y_test,pred))#L'affichage de la matrice de confusion.
print('Acurracy is',accuracy_score(y_test,pred))# affichage de l'accuracy
print(classification_report(y_test,pred))#l'affichage du rappot de classification

# KNN
model = KNeighborsClassifier()# Initialization de modéle.
model.fit(x_train,y_train)# l'entrainment du modéle.
pred = model.predict(x_test)# La prédiction dans les données de Test
print('\n Les performances de \'k plus proches voisins\''+10*'-')
print(confusion_matrix(y_test,pred))#L'affichage de la matrice de confusion.
print('Acurracy is',accuracy_score(y_test,pred))# affichage de l'accuracy
print(classification_report(y_test,pred))#l'affichage du rappot de classification

# GaussianNB
model = GaussianNB()# Initialization de modéle.
model.fit(x_train,y_train)# l'entrainment du modéle.
pred = model.predict(x_test)# La prédiction dans les données de Test
print('\n Les performances de \'Naive bayes\''+10*'-')
print(confusion_matrix(y_test,pred))#L'affichage de la matrice de confusion.
print('Acurracy is',accuracy_score(y_test,pred))# affichage de l'accuracy
print(classification_report(y_test,pred))#l'affichage du rappot de classification

