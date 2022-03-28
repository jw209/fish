# code here
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from statistical_summary import statistical_summary

fish_data = pd.read_csv('Fish.csv')

plt.plot(fish_data["Weight"], fish_data["Height"], 'ro')
plt.ylabel('Height')
plt.xlabel('Weight')

statistical_summary(fish_data)

plt.show()
