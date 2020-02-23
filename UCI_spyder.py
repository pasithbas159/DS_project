import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 1. importing
mat = pd.read_csv('C:/jupyterout/UCI_project/student-mat.csv', sep = ';')
# 2. cleaning&formating
cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'reason', 'guardian', 'studytime',
       'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc'
       , 'Walc', 'health']
for i in cols:
    mat[i] = mat[i].astype('category')
# =============================================================================
# # 3. visualization&EDA
# print(mat.loc[:, mat.dtypes == np.int64].corr())
# sns.heatmap(mat.loc[:, mat.dtypes == np.int64])
# sns.pairplot(mat.loc[:, mat.dtypes == np.int64])
# # age, school and sex
# plt.figure(figsize = (8, 5))
# plt.subplot(131)
# sns.countplot(x=mat.school)
# plt.subplot(132)
# sns.countplot(x=mat.age)
# plt.subplot(133)
# sns.countplot(x=mat.sex)
# plt.show()
# # =============================================================================
# # # absences VS Grade
# # plt.subplot(131)
# # sns.lmplot(x='absences', y='G1', data=mat)
# # plt.subplot(132)
# # sns.lmplot(x='absences', y='G2', data=mat)
# # plt.subplot(133)
# # sns.lmplot(x='absences', y='G3', data=mat)
# # plt.show()
# # =============================================================================
# sns.lmplot(x='G1', y='G2', data=mat, col='Dalc')
# sns.lmplot(x='G2', y='G3', data=mat, col='Dalc')
# =============================================================================
# 4. train_test_split
from sklearn.model_selection import train_test_split
X = mat[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2']]
y = mat['G3']
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
from sklearn.metrics import confusion_matrix, accuracy_score
print(y_pred)
print('-'*100)
print(confusion_matrix(y_test, y_pred))
print(f'R-Squared: {pipeline.score(X_train, y_train)}') #0.188
print(f'accuracy score: {accuracy_score(y_test, y_pred)}') #0.101

# =============================================================================
# # Special: statsmodels
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# cols_one = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
#         'reason', 'guardian', 'studytime',
#        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
#        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc'
#        , 'Walc', 'health']
# dat = pd.concat([mat, pd.get_dummies(mat[cols_one], drop_first=True)], axis = 1).drop(columns=(cols_one))
# dat = dat.drop(columns=['Fjob', 'Mjob'])
# train_dat, test_dat = train_test_split(dat, train_size=0.8, random_state=101)
# formula = ('G3 ~ age + traveltime + failures + absences + G1 + G2 + school_MS + sex_M + address_U + famsize_LE3 + Pstatus_T + Medu_1 + Medu_2 + Medu_3 + Medu_4 + Fedu_1 + Fedu_2 + Fedu_3 + Fedu_4 + reason_home + reason_other + reason_reputation + guardian_mother + guardian_other + studytime_2 + studytime_3 + studytime_4 + schoolsup_yes + famsup_yes + paid_yes + activities_yes + nursery_yes + higher_yes + internet_yes + romantic_yes + famrel_2 + famrel_3 + famrel_4 + famrel_5 + freetime_2 + freetime_3 + freetime_4 + freetime_5 + goout_2 + goout_3 + goout_4 + goout_5+ Dalc_2 + Dalc_3 + Dalc_4 + Dalc_5 + Walc_2 + Walc_3 + Walc_4 + Walc_5 + health_2 + health_3 + health_4 + health_5')
# model_a = smf.ols(formula=formula, data=train_dat).fit()
# print(model_a.summary())
# =============================================================================
