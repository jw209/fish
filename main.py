# code here
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


fish_data = pd.read_csv('Fish.csv')


# function to plot the data
def show_graphs():
    # pair plot
    sns.pairplot(fish_data, hue='Species')
    plt.figure(figsize=(12, 8))
    plt.show()
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


show_graphs()
