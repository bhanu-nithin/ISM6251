import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
#import os
#exit(os.getcwd())

winning_model = pickle.load(open('C:/Users/Nithin Yadav/Desktop/DSP/winning_model.pkl', "wb"))

print("\n*****************************************************")
print("* Prediction model for lawnmower *")
print("*****************************************************\n")
income = float(input("Enter the income of the person: "))
lot_size = float(input("enter the lot size value "))
df = pd.DataFrame({'income': [income]},{'lot_size':[lot_size]})
result = winning_model.predict(df)
probability = winning_model.predict_proba(df)
Ownership = ('no ownership', 'ownership')
print(f"\n The prediction model for the lawnmower is at {probability[0][1]:.4f}, therefore it's indicated that  {Ownership[result[0]]}.\n")