# code here
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statistical_summary import statistical_summary
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

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
    # pie-chart of number of species
    plt.pie(fish_data['Species'].value_counts(), labels=fish_data['Species'].value_counts().index, autopct='%1.1f%%')
    plt.show()


statistical_summary(fish_data)

show_graphs()


# split data into training and testing sets
X = fish_data.drop('Species', axis=1)
y = fish_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# instantiate the model
log_reg = LogisticRegression()
# fit the model
log_reg.fit(X_train, y_train)
# predict the response
y_pred = log_reg.predict(X_test)
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# accuracy score
print(accuracy_score(y_test, y_pred))
