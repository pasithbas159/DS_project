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
#test