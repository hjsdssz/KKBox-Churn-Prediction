import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

train = pd.read_csv('train_v2.csv')
sample_submission = pd.read_csv('sample_submission_v2.csv')
transactions = pd.read_csv('transactions_v2.csv')
user_logs = pd.read_csv('user_logs_v2.csv')
members = pd.read_csv('members_v3.csv')

# set the options so the output format can be displayed correctly
pd.set_option('expand_frame_repr', True)
pd.set_option('display.max_rows', 30000000)
pd.set_option('display.max_columns', 100)

# check the number of duplicate accounts in each table
train.duplicated('msno').sum()
sample_submission.duplicated('msno').sum()
transactions.duplicated('msno').sum()
user_logs.duplicated('msno').sum()
members.duplicated('msno').sum()

# returns the max value of numerical variables and membership_expire_date
# returns the min value of transaction date
# returns the mode of ordinal variable and dummy variables, if multiple values share the same frequency, keep the first one
transactions_v2 = transactions.groupby('msno', as_index = False).agg({'payment_method_id': lambda x:x.value_counts().index[0], 'payment_plan_days': 'max', 'plan_list_price': 'max',
                                       'actual_amount_paid': 'max', 'is_auto_renew': lambda x:x.value_counts().index[0], 'transaction_date': 'min', 'membership_expire_date': 'max',
                                       'is_cancel': lambda x:x.value_counts().index[0]})

# returns the max value of date and number of unique songs
# returns the sum of other variables
user_logs_v2 = user_logs.groupby('msno', as_index = False).agg({'date': 'max', 'num_25': 'sum', 'num_50': 'sum', 'num_75': 'sum',
                                 'num_985': 'sum', 'num_100': 'sum', 'num_unq': 'max', 'total_secs': 'sum'})

# calculate the percentage of number of songs played within certain period
user_logs_v2['percent_25'] = user_logs_v2['num_25']/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])
user_logs_v2['percent_50'] = user_logs_v2['num_50']/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])
user_logs_v2['percent_100'] = (user_logs_v2['num_985']+user_logs_v2['num_100'])/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])

# drop useless variables
user_logs_v3 = user_logs_v2.drop(columns = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100'])

# merge between different tables for modelling purpose
dataset_train = train.merge(members, on = 'msno', how = 'left').merge(transactions_v2, on = 'msno', how = 'left').merge(user_logs_v3, on = 'msno', how = 'left')
dataset_train.dtypes

# date in csv will be recognized as float in python
# this value needs to be converted back to date
dataset_train['registration_init_time'] = pd.to_datetime(dataset_train['registration_init_time'], format = '%Y%m%d')
dataset_train['transaction_date'] = pd.to_datetime(dataset_train['transaction_date'], format = '%Y%m%d')
dataset_train['membership_expire_date'] = pd.to_datetime(dataset_train['membership_expire_date'], format = '%Y%m%d')
dataset_train['date'] = pd.to_datetime(dataset_train['date'], format = '%Y%m%d')

# check the maximum of datetime value
dataset_train.select_dtypes(include = ['datetime64[ns]']).max()

# create new day columns for modelling purpose
dataset_train['registration_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['registration_init_time']).astype('timedelta64[D]')
dataset_train['transaction_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['transaction_date']).astype('timedelta64[D]')
dataset_train['membership_expire_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['membership_expire_date']).astype('timedelta64[D]')
dataset_train['last_play_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['date']).astype('timedelta64[D]')

# check the distribution of age due to the documentation explanation
dataset_train['bd'].value_counts()

# remove gender and age since missing value or incorrect value is over 50%
dataset_train_v2 = dataset_train.drop(columns = ['msno', 'gender', 'bd', 'registration_init_time', 'transaction_date', 'membership_expire_date', 'date'])
dataset_train_v2.dtypes

# check the number of missing values in each column
dataset_train_v2.isna().sum()

# Handle missing value of part of numeric columns by using mode
def replacemode(i):
    dataset_train_v2[i] = dataset_train_v2[i].fillna(dataset_train_v2[i].value_counts().index[0])
    return 

replacemode('city')
replacemode('registered_via')
replacemode('payment_method_id')
replacemode('payment_plan_days')
replacemode('is_auto_renew')
replacemode('is_cancel')

# Handle missing value of part of numeric columns by using mean
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
def replacemean(i):
    dataset_train_v2[i] = mean_imputer.fit_transform(dataset_train_v2[[i]])
    return 

replacemean('plan_list_price')
replacemean('actual_amount_paid')
replacemean('num_unq')
replacemean('total_secs')
replacemean('percent_25')
replacemean('percent_50')
replacemean('percent_100')
replacemean('registration_day')
replacemean('transaction_day')
replacemean('membership_expire_day')
replacemean('last_play_day')

# Handle outliers by using capping
def replaceoutlier(i):
    mean, std = np.mean(dataset_train_v2[i]), np.std(dataset_train_v2[i])
    cut_off = std*3
    lower, upper = mean - cut_off, mean + cut_off
    dataset_train_v2[i][dataset_train_v2[i] < lower] = lower
    dataset_train_v2[i][dataset_train_v2[i] > upper] = upper
    return

replaceoutlier('plan_list_price')
replaceoutlier('actual_amount_paid')
replaceoutlier('num_unq')
replaceoutlier('total_secs')
replaceoutlier('percent_25')
replaceoutlier('percent_50')
replaceoutlier('percent_100')
replaceoutlier('registration_day')
replaceoutlier('transaction_day')
replaceoutlier('membership_expire_day')
replaceoutlier('last_play_day')

dataset_train_v2.dtypes
dataset_train_v2.describe()

# convert categorical variables into string for get_dummies
dataset_train_v2.iloc[:, 1:4] = dataset_train_v2.iloc[:, 1:4].astype(str)

# create dummy variables for modelling purpose
dataset_train_v3 = pd.get_dummies(dataset_train_v2, drop_first = True)
dataset_train_v3.dtypes

# Feature Scaling for modelling purpose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X = dataset_train_v3.drop(columns = ['is_churn'])
Y = dataset_train_v3['is_churn']
nm_X = pd.DataFrame(MinMaxScaler().fit_transform(X))
nm_X.columns = X.columns.values
nm_X.index = X.index.values
sc_X = pd.DataFrame(StandardScaler().fit_transform(X))
sc_X.columns = X.columns.values
sc_X.index = X.index.values

# Visualize the correlation between independent columns
sn.set(style="white")
corr = nm_X.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

sn.set(style="white")
corr2 = sc_X.corr()
mask = np.zeros_like(corr2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Remove those columns with high correlation values
nm_X_v2 = nm_X.drop(columns = ['membership_expire_day', 'percent_100', 'registered_via_9.0', 'payment_method_id_38.0'])
sc_X_v2 = sc_X.drop(columns = ['membership_expire_day', 'percent_100', 'registered_via_9.0', 'payment_method_id_38.0'])

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif
nm_col = ['payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'num_unq', 'total_secs', 'percent_25', 'percent_50', 'registration_day',
          'transaction_day', 'last_play_day']
nm_X_v3 = nm_X_v2.drop(columns = nm_col)
nm_X_v4 = pd.DataFrame(nm_X_v2, columns = nm_col)
nm_X_v5 = pd.DataFrame(SelectKBest(score_func=chi2, k='all').fit(nm_X_v3, Y).pvalues_ <= 0.05, columns = ['importance'])
nm_X_v5.index = nm_X_v3.columns.values
nm_X_v6 = pd.DataFrame(SelectKBest(score_func=f_classif, k='all').fit(nm_X_v4, Y).pvalues_ <= 0.05, columns = ['importance'])
nm_X_v6.index = nm_X_v4.columns.values
nm_X_v7 = pd.concat([nm_X_v5, nm_X_v6])
nm_selected = list(pd.Series(nm_X_v7[nm_X_v7['importance'] == 1].index.values))
nm_X_v8 = pd.DataFrame(nm_X_v2, columns = nm_selected)
sc_X_v3 = pd.DataFrame(SelectKBest(score_func=f_classif, k='all').fit(sc_X_v2, Y).pvalues_ <= 0.05, columns = ['importance'])
sc_X_v3.index = sc_X_v2.columns.values
sc_selected = list(pd.Series(sc_X_v3[sc_X_v3['importance'] == 1].index.values))
sc_X_v4 = pd.DataFrame(sc_X_v2, columns = sc_selected)

# Reduce Dimension since we still have too many features on standardized results
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(sc_X_v4)
np.cumsum(pca.explained_variance_ratio_)
sc_X_v5 = PCA(n_components=50).fit_transform(sc_X_v4)
pca_v2 = PCA()
pca_v2.fit_transform(nm_X_v8)
np.cumsum(pca_v2.explained_variance_ratio_)
nm_X_v9 = PCA(n_components=25).fit_transform(nm_X_v8)

# Split into train and test Set
from sklearn.model_selection import train_test_split
nm_X_train, nm_X_test, nm_Y_train, nm_Y_test = train_test_split(nm_X_v9, Y, test_size = 0.3, random_state = 0)
sc_X_train, sc_X_test, sc_Y_train, sc_Y_test = train_test_split(sc_X_v5, Y, test_size = 0.3, random_state = 0)

# Fit training set into different algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

nm_models = []
nm_models.append(('KNN', KNeighborsClassifier()))
nm_models.append(('LR', LogisticRegression()))
nm_models.append(('LDA', LinearDiscriminantAnalysis()))
nm_models.append(('QDA', QuadraticDiscriminantAnalysis()))
nm_models.append(('CART', DecisionTreeClassifier()))
nm_models.append(('NB', GaussianNB()))
nm_models.append(('Linear SVM', SVC(kernel = 'linear')))
nm_models.append(('Kernel SVM', SVC(kernel = 'rbf')))
sc_models = []
sc_models.append(('KNN', KNeighborsClassifier()))
sc_models.append(('LR', LogisticRegression()))
sc_models.append(('LDA', LinearDiscriminantAnalysis()))
sc_models.append(('QDA', QuadraticDiscriminantAnalysis()))
sc_models.append(('CART', DecisionTreeClassifier()))
sc_models.append(('NB', GaussianNB()))
sc_models.append(('Linear SVM', SVC(kernel = 'linear')))
sc_models.append(('Kernel SVM', SVC(kernel = 'rbf')))
ensembles = []
ensembles.append(('BC', BaggingClassifier(base_estimator=DecisionTreeClassifier())))
ensembles.append(('AB', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('XGB', XGBClassifier()))

from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in nm_models:
	nm_cv_results = cross_val_score(model, nm_X_train, nm_Y_train, cv=10, scoring='f1', n_jobs = -1)
	results.append(nm_cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, nm_cv_results.mean(), nm_cv_results.std())
	print(msg)

results2 = []
names2 = []
for name2, model2 in sc_models:
	sc_cv_results = cross_val_score(model2, sc_X_train, sc_Y_train, cv=10, scoring='f1', n_jobs = -1)
	results2.append(sc_cv_results)
	names2.append(name2)
	msg2 = "%s: %f (%f)" % (name2, sc_cv_results.mean(), sc_cv_results.std())
	print(msg2)

results3 = []
names3 = []
for name3, model3 in ensembles:
	en_cv_results = cross_val_score(model3, sc_X_train, sc_Y_train, cv=10, scoring='f1', n_jobs = -1)
	results3.append(en_cv_results)
	names3.append(name3)
	msg3 = "%s: %f (%f)" % (name3, en_cv_results.mean(), en_cv_results.std())
	print(msg3)
    
results4 = []
names4 = []
for name4, model4 in ensembles:
	en_cv_results2 = cross_val_score(model4, nm_X_train, nm_Y_train, cv=10, scoring='f1', n_jobs = -1)
	results4.append(en_cv_results2)
	names4.append(name4)
	msg4 = "%s: %f (%f)" % (name4, en_cv_results2.mean(), en_cv_results2.std())
	print(msg4)

# Apply Grid Search on Random Forest since it returns the best result on Cross Validation among all models
from sklearn.model_selection import GridSearchCV
parameters = {"max_depth": [20],
              "max_features": [15],
              "min_samples_split": [15],
              "min_samples_leaf": [5],
              "criterion": ["entropy"]}
grid_search_RF = GridSearchCV(estimator = RandomForestClassifier(), param_grid = parameters, scoring = "f1", cv = 10, n_jobs = -1)
grid_result_RF = grid_search_RF.fit(nm_X_train, nm_Y_train)
print("Best: %f using %s" % (grid_result_RF.best_score_, grid_result_RF.best_params_))
# max_depth - 20 / max_features - 15 / min_samples_leaf - 15 / min_samples_split - 5

# Apply Grid Search on Bagging since it returns the second best result on Cross Validation among all models
parameters2 = {"max_samples": [0.8],
              "max_features": [0.8]}
grid_search_BC = GridSearchCV(estimator = BaggingClassifier(base_estimator=DecisionTreeClassifier()), param_grid = parameters2, scoring = "f1", cv = 10, n_jobs = -1)
grid_result_BC = grid_search_BC.fit(nm_X_train, nm_Y_train)
print("Best: %f using %s" % (grid_result_BC.best_score_, grid_result_BC.best_params_))
# max_samples - 0.8 / max_features - 0.8

# Evaluate tuned random forest model result on test dataset because it provides the best result
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
nm_Y_predict = grid_result_RF.predict(nm_X_test)
acc = accuracy_score(nm_Y_test, nm_Y_predict)
prec = precision_score(nm_Y_test, nm_Y_predict)
rec = recall_score(nm_Y_test, nm_Y_predict)
f1 = f1_score(nm_Y_test, nm_Y_predict)  
model_results = pd.DataFrame([['Random Forest', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
# F1 Score - 83.3%

# Check the important features based on random forest results
RF_model = RandomForestClassifier(max_depth=20,max_features=15,min_samples_split=15,min_samples_leaf=5,criterion='entropy')
RF_result = RF_model.fit(nm_X_train, nm_Y_train)
RF_result.feature_importances_