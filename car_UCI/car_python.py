# 1. importing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\project\car_UCI\car_data.csv', sep=',', 
                 names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'], index_col=False)
df = pd.DataFrame(df)
# 2. formatting
cols = ['buying', 'maint', 'lug_boot', 'safety', 'class']
for i in cols:
    df[i] = df[i].astype('category')
for i in cols:
    print(df[i].cat.categories)

df.doors = df.doors.str.replace('5more', '6')
df.persons = df.persons.str.replace('more', '5')

df.doors = df.doors.astype('int32')
df.persons = df.persons.astype('int32')

from sklearn.preprocessing import OneHotEncoder
trans_cols = ['buying', 'maint', 'lug_boot', 'safety']
ohenc=OneHotEncoder(sparse=False, drop='first')
m3=ohenc.fit_transform(df[trans_cols])

m3 = pd.DataFrame(m3).rename(columns={0:'buying _low', 1:'buying _med', 2:'buying _vhigh', 
                                      3:' maint _low', 4:' maint _med', 5:' maint _vhigh', 
                                      6:'lug_boot_med', 7:'lug_boot_small', 8:'safety_low', 
                                      9:'safety_med'}).astype('int32')
dat = pd.concat([df, m3], axis = 1).drop(columns=(trans_cols))
# 3. Visualization&EDA
# sns.pairplot(dat)
# sns.countplot(x=dat['doors'])
# sns.countplot(x=dat['persons'])
# 4. train_test_split
from sklearn.model_selection import train_test_split
X = dat[[ 'doors', 'buying _low', 'buying _med',
       'buying _vhigh', ' maint _low', ' maint _med', ' maint _vhigh',
       'lug_boot_med', 'lug_boot_small', 'safety_low', 'safety_med']]
y = dat[['class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
# 5. modeling
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
model.score(X_train, y_train)
# 6. predicted
predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))