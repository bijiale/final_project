import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train=pd.read_csv(r"E:\data\titanic\train.csv")
#fig1=plt.figure()
#plt.subplot(4,1,1)
#data_train.groupby('Survived')['Survived'].count().plot(kind='bar')
#plt.subplot(4,1,2)
#data_train.groupby('Pclass')['Pclass'].count().plot(kind='bar')
#plt.subplot(4,1,3)
#plt.subplot2grid((2,3),(1,0), colspan=2)
#data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
#data_train.Age[data_train.Pclass == 2].plot(kind='kde')
#data_train.Age[data_train.Pclass == 3].plot(kind='kde')
#plt.legend(('Class1','Class2','Class3'))

#变量对年龄的影响

#将sibsp和parch合并为family
family=data_train['SibSp']+data_train['Parch']
data_train.insert(0,'family',family)
print(data_train.corr()['Age'])
data_train.loc[(data_train['family']>0),'is_family']=1
data_train.loc[(data_train['family']==0),'is_family']=0
#family_survived=data_train.groupby(['family','Survived'])['Survived'].count().unstack()
#family_survived.plot(kind='bar',stacked=True)
print(data_train.corr()['Age'])



df=data_train[['Age','Pclass','Fare','family']]

from sklearn.ensemble import RandomForestRegressor
def set_missing_age(df):
    known_age=df[df['Age'].notnull()].values
    unknown_age=df[df['Age'].isnull()].values
    y=known_age[:,0]
    x=known_age[:,1:]
    rfr=RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x,y)
    predictedAges=rfr.predict(unknown_age[:,1:])
    df.loc[(df['Age'].isnull()),'Age']=predictedAges
    return df,rfr
full_df,rfr=set_missing_age(df)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
age=full_df['Age'].values.reshape(-1,1)
age_param=scaler.fit(age)
data_train['Age_scaler']=pd.DataFrame(scaler.fit_transform(age,age_param))
fare=full_df['Fare'].values.reshape(-1,1) 
fare_param=scaler.fit(fare)
data_train['Fare_scaler']=pd.DataFrame(scaler.fit_transform(fare,fare_param))

dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
dummies_family=pd.get_dummies(data_train['is_family'],prefix='is_family')
df_train = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_family], axis=1)
df_train.drop(['PassengerId','Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','family','Age','Fare','SibSp','Parch','is_family'],inplace=True,axis=1)

#逻辑回归建模
from sklearn.linear_model import LogisticRegression
train_np=df_train.values
y=train_np[:,0]
x=train_np[:,1:]
clf=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x,y)

data_test=pd.read_csv(r"E:\data\titanic\test.csv")
family=data_test['SibSp']+data_test['Parch']
data_test.insert(0,'family',family)
data_test.loc[(data_test['family']>0),'is_family']=1
data_test.loc[(data_test['family']==0),'is_family']=0
df_test=data_test[['Age','Fare','Pclass','family']]
df_test.loc[(df_test['Fare'].isnull()),'Fare']=0
df,_=set_missing_age(df_test)
dummies_family=pd.get_dummies(data_test['is_family'],prefix='is_family')
dummies_Pclass=pd.get_dummies(data_test['Pclass'],prefix='Pclass')
dummies_Embarked=pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
df_test = pd.concat([data_test, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_family], axis=1)
df_test.drop(['PassengerId','Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','family','is_family','SibSp','Parch'], axis=1, inplace=True)
age=df['Age'].values.reshape(-1,1)
df_test['Age'] = pd.DataFrame(scaler.fit_transform(age, age_param))
fare=df['Fare'].values.reshape(-1,1)
df_test['Fare'] = pd.DataFrame(scaler.fit_transform(fare, fare_param))
predictions = clf.predict(df_test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#result.to_csv(r"E:\data\titanic\logistic_regression_predictions.csv", index=False)

from sklearn.model_selection import train_test_split
split_train,split_test=train_test_split(df_train,test_size=0.3)
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(split_train.values[:,:1],split_train.values[:,0])
predictions=clf.predict(split_test.values[:,1:])
split_test.insert(0,'pre',predictions)
