#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  

x = np.arange(len(people))  

plt.bar(x, fruit[0], color=colors[0], width=0.5, label=fruit_names[0])
plt.bar(x, fruit[1], bottom=fruit[0], color=colors[1], width=0.5, label=fruit_names[1])
plt.bar(x, fruit[2], bottom=fruit[0]+fruit[1], color=colors[2], width=0.5, label=fruit_names[2])
plt.bar(x, fruit[3], bottom=fruit[0]+fruit[1]+fruit[2], color=colors[3], width=0.5, label=fruit_names[3])

plt.xticks(x, people)
plt.ylabel("Quantity of Fruit")
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.title("Number of Fruit per Person")
plt.legend()

plt.show()
