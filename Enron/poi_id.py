#!/usr/bin/python
# coding: utf-8

# In[1]:


import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from time import time


# In[2]:


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print '--------feature number-------------'
print len(data_dict),'\n'
poi_count = 0
for person in data_dict.values():
    if person['poi'] == True:
        poi_count += 1
print '--------count poi------------------'
print 'count poi: ', poi_count,'\n'
features_list = ['poi', 'salary', 'bonus']
all_features = data_dict.values()[0].keys()
print '------number of all_features-------'
print 'number of all_features: ', len(all_features),'\n'
d = {}
for feature in all_features:
    d[feature] = 0
    for value in data_dict.values():
        if value[feature] == 'NaN':
            d[feature] += 1
print '---------NaN count---------'
print d,'\n'

# 临时排除poi，并清除无效特征
f1 = all_features
all_features.remove('email_address')
all_features.remove('poi')
all_features.remove('loan_advances')
all_features.remove('director_fees')


# In[4]:

print '--------useless name--------'
for name, value in data_dict.items():
    count = 0
    for i in f1:
        if value[i] == 'NaN':
            count += 1
    if count == len(f1):
        print name

print '\n'
# Task 2: Remove outliers
# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.


data_dict.pop('LOCKHART EUGENE E')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


def draw(data):
    for point in data:
        salary = point['salary']
        bonus = point['bonus']
        pyplot.scatter(salary, bonus)
    pyplot.xlabel('salary')
    pyplot.ylabel('bonus')
    pyplot.show()


draw(data_dict.values())
data_dict.pop('TOTAL')
draw(data_dict.values())


# 添加新特征


def computeFraction(poi_messages, all_messages):
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages / float(all_messages)
    else:
        fraction = 0
    return fraction


my_dataset = data_dict
for name, feature in my_dataset.items():
    my_dataset[name]['fraction_from_poi'] = computeFraction(
        my_dataset[name]['from_poi_to_this_person'], my_dataset[name]['to_messages'])
    my_dataset[name]['fraction_to_poi'] = computeFraction(
        my_dataset[name]['from_this_person_to_poi'], my_dataset[name]['from_messages'])


all_features.insert(0, 'poi')
all_features.insert(1, 'fraction_from_poi')
all_features.insert(1, 'fraction_to_poi')
print '--------all feature---------'
print all_features,'\n'


# 选择最佳特征


data_array = featureFormat(my_dataset, all_features, sort_keys=True)
labels, features = targetFeatureSplit(data_array)
selector = SelectKBest(k='all')
selector.fit(features, labels)
scores = selector.get_params()
tuples = zip(all_features[1:], scores)
k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)
print '--------------k_best_features-------------'
print 'k_best_features: ', k_best_features, '\n'


# In[16]:


my_features_list = [i[0] for i in k_best_features[:5]]
if 'poi' not in my_features_list:
    my_features_list.insert(0, 'poi')
print '------------my_features_list--------------'
print 'my_features_list',my_features_list,'\n'


# In[17]:


data_array = featureFormat(my_dataset, my_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data_array)
scaler = MinMaxScaler()
features_new = scaler.fit_transform(features)


# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print '--------------GridSearchCV---------------'

labels_train, labels_test, features_train, features_test = train_test_split(
    labels, features_new, test_size=0.3, random_state=42)
t1 = time()
clf_gnb = GaussianNB()
parm = {}
pipline = Pipeline([('scaler', scaler), ('gnb', clf_gnb)])
grid = GridSearchCV(pipline, parm)
grid.fit(features_train, labels_train)
clf_gnb = grid.best_estimator_
print "\nGaussianNB score: ", clf_gnb.score(features_test, labels_test)
print "\nGaussianNB test: ", test_classifier(clf_gnb, my_dataset, my_features_list)
print '\nGaussianNB time', time() - t1


# In[20]:

t2 = time()
clf_dtc = tree.DecisionTreeClassifier()
parms = {'max_depth': range(5, 15), 'min_samples_leaf': range(1, 5)}
grid = GridSearchCV(clf_dtc, parms)
grid.fit(features_train, labels_train)
clf_dtc = grid.best_estimator_
print "\nDecision Tree Classifier score: ", clf_dtc.score(features_test, labels_test)
print "\nDecision Tree Classifier test: ", test_classifier(clf_dtc, my_dataset, my_features_list)
print "\nDecision Tree Classifier:", time() - t2


# In[21]:


t3 = time()
clf_svc = SVC()
parms = {'svc__kernel': ('linear', 'rbf', 'poly',
                         'sigmoid'), 'svc__C': range(1, 10)}
pipeline = Pipeline([('scaler', scaler), ('svc', clf_svc)])
grid = GridSearchCV(pipeline, parms)
grid.fit(features_train, labels_train)
clf_svc = grid.best_estimator_
print "\nSVM: ", clf_svc.score(features_train, labels_train)
print "\nSVC test: ", test_classifier(clf_svc, my_dataset, my_features_list)
print "\nSVC time: ", time() - t3


# In[22]:

t4=time()
clf_rfc=RandomForestClassifier()
parms = {'n_estimators': range(2, 5), 'min_samples_split': range(2, 5), 'max_depth': range(2, 15), 
                'min_samples_leaf': range(1, 5), 'random_state': [0, 10, 23, 36, 42], 'criterion': ['entropy', 'gini']}
grid = GridSearchCV(clf_rfc, parms)
grid.fit(features_train,labels_train)
clf_rfc = grid.best_estimator_
print "\nRandomForestClassifier: ",clf_rfc.score(features_test,labels_test)
print "\nRandomForestClassifier test: ", test_classifier(clf_rfc,my_dataset,my_features_list)
print "\nRandomForestClassifier time: ", time()-t4

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

clf = tree.DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=4, min_samples_split=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print '-----------------'
print 'feature_importances_', clf.feature_importances_
print '-----------------'
print 'score', clf.score(features_test, labels_test)
print '-----------------'
print metrics.precision_score(labels_test, pred)
dump_classifier_and_data(clf, my_dataset, my_features_list)


print '--------------test features number-------------------'

for i in range(2, 4):
    print my_features_list[:i]
    data = featureFormat(my_dataset, my_features_list[:i], sort_keys = True,remove_all_zeroes=True)
    labels, features = targetFeatureSplit(data)
    clf = tree.DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=4, min_samples_split=2)
    clf.fit(features,labels)
    test_classifier(clf,my_dataset,my_features_list[:i])
