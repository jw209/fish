# code here
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

fish_data = pd.read_csv('Fish.csv')

plt.plot(fish_data["Weight"], fish_data["Height"], 'ro')
plt.ylabel('Height')
plt.xlabel('Weight')

plt.show()
