# code here
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statistical_summary import statistical_summary
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import plotly.express as px

matplotlib.use("TkAgg")

fish_data = pd.read_csv('Fish.csv')


# function to plot the data
def show_graphs():
    # pair plot
    sns.pairplot(fish_data, hue='Species')
    plt.show()
    plt.clf()
    # weight against height
    plt.plot(fish_data["Weight"], fish_data["Height"], 'ro')
    plt.ylabel('Height')
    plt.xlabel('Weight')
    plt.show()
    # boxplot of Length1 by Species
    sns.boxplot(x='Species', y='Length1', data=fish_data)
    plt.show()
    # scatter plot of Length1 by height and wieght
    px.scatter(fish_data, x='Height', y='Length1', size='Weight', color='Species')
    # pie-chart of number of species
    plt.pie(fish_data['Species'].value_counts(), labels=fish_data['Species'].value_counts().index, autopct='%1.1f%%')
    plt.show()


statistical_summary(fish_data)

#show_graphs()

# prepare data for models
# split data into training and testing sets
X = fish_data.drop('Species', axis=1)
y = fish_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# this part of the code is related to the logistic regression model
# instantiate the logistic regression model
log_reg = LogisticRegression(C=10)
# fit the model
log_reg.fit(X_train, y_train)
# predict the response
y_pred = log_reg.predict(X_test)
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# testing accuracy score
print(accuracy_score(y_test, y_pred))
# training accuracy score
print("Test set score: {:.2f}".format(log_reg.score(X_test, y_test)))
print("Training set score: {:.2f}".format(log_reg.score(X_train, y_train)))


# this part of the code is related to the SVM model
# instantiate the logistic regression model
svmModel = svm.SVC(kernel='linear')
svmModel.fit(X_train, y_train)

print("svm support vectors: {}".format(svmModel.n_support_))
print("svmModel support vector indices: {}".format(svmModel.support_))
print("svmModel # of support vectors in each class: {}".format(svmModel.n_support_))

print("Training set score: {:.2f}".format(svmModel.score(X_train, y_train)))
print("Test set score: {:.2f}".format(svmModel.score(X_test, y_test)))

cm = confusion_matrix(y_test, svmModel.predict(X_test))
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

'''
GRID = [
    {'scaler': [StandardScaler()],
     'estimator': [MLPClassifier(random_state=0)],
     'estimator__solver': ['adam', 'sgd'],
     'estimator__learning_rate_init': [0.0015],
     'estimator__max_iter': [7000],
     'estimator__hidden_layer_sizes': [(20, 4), (5, 2), (10, 2), (10, 10), (7, 8), (10, 16), (2, 5), (10, 7), (10, 4)],
     'estimator__activation': ['relu', 'tanh'],
     'estimator__alpha': [1e-5, 1e-4],
     'estimator__early_stopping': [True, False],
     'estimator__warm_start': [True],
     }
]

#import the grid search object
from sklearn.model_selection import GridSearchCV
# import the pipeline object
from sklearn.pipeline import Pipeline
# import make_scorer
from sklearn.metrics import make_scorer

PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])

clf = GridSearchCV(estimator=PIPELINE, param_grid=GRID,
                            scoring=make_scorer(accuracy_score),# average='macro'),
                            n_jobs=-1, cv=3, refit=True, verbose=1,
                            return_train_score=False)
'''

clf = MLPClassifier(random_state=0, solver='adam', max_iter=7000, hidden_layer_sizes=(20, 4), beta_1=0.8, beta_2=0.90, epsilon=1e-20, alpha=0.00001)
clf.fit(X_train, y_train)

#print("Best parameters: {}".format(clf.best_params_))
#print("Best score: {:.2f}".format(clf.best_score_))

print("Neural network model training accuracy: {:.2f}".format(clf.score(X_train, y_train)))
print("Neural network model testing accuracy: {:.2f}".format(clf.score(X_test, y_test)))

cm = confusion_matrix(y_test, clf.predict(X_test))
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()


