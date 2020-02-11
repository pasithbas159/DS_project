import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 1. importing
mat = pd.read_csv('C:/jupyterout/UCI_project/student-mat.csv', sep = ';')
por = pd.read_csv('C:/jupyterout/UCI_project/student-por.csv', sep = ';')
# 2. cleaning&formating
cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'reason', 'guardian', 'studytime',
       'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc'
       , 'Walc', 'health']
for i in cols:
    mat[i] = mat[i].astype('category')
    por[i] = por[i].astype('category')
# 3. visualization&EDA
print(mat.loc[:, mat.dtypes == np.int64].corr())
sns.heatmap(mat.loc[:, mat.dtypes == np.int64])
sns.pairplot(mat.loc[:, mat.dtypes == np.int64])
# age, school and sex
plt.figure(figsize = (8, 5))
plt.subplot(131)
sns.countplot(x=mat.school)
plt.subplot(132)
sns.countplot(x=mat.age)
plt.subplot(133)
sns.countplot(x=mat.sex)
plt.show()
# =============================================================================
# # absences VS Grade
# plt.subplot(131)
# sns.lmplot(x='absences', y='G1', data=mat)
# plt.subplot(132)
# sns.lmplot(x='absences', y='G2', data=mat)
# plt.subplot(133)
# sns.lmplot(x='absences', y='G3', data=mat)
# plt.show()
# =============================================================================
sns.lmplot(x='G1', y='G2', data=mat, col='Dalc')
sns.lmplot(x='G2', y='G3', data=mat, col='Dalc')
# 4. train_test_split
from sklearn.model_selection import train_test_split
X = mat[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences']]
y = mat['G1']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
# 5. pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
col_trans = make_column_transformer(
        (OneHotEncoder(), ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'reason', 'guardian', 'studytime',
       'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc'
       , 'Walc', 'health']))
col_trans

from sklearn.linear_model import LinearRegression
model = LinearRegression()
pipeline = make_pipeline(col_trans, model)
pipeline
# 6. modeling
pipeline.fit(X_train, y_train)
pipeline.score(X_train, y_train)
y_pred = pipeline.predict(X_test).round(0)
# 7. test accuracy
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print('-'*100)
print(classification_report(y_test, y_pred))