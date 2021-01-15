import sys
import pandas as pd
import numpy as np
import pickle

'''
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
ocean_proximity
'''


args=sys.argv
if len(args)!=8:
    print("Usage:")
    print("Python main.py <housing_median_ag< <total_rooms> <total_bedrooms> < population><households><median_income> <ocean_proximity>")
    exit()

print(args)
housing_median_age = float(args[1])
total_rooms = float(args[2])
total_bedrooms = float(args[3])
population = float(args[4])
households = float(args[5])
median_income = float(args[6])
ocean_proximity = float(args[7])

X = np.array([[housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity]])
#load model from saved file_extension
f=open("LR_without_scaling.sav","rb")
model=pickle.load(f)
f.close()

print(model.predict(X))
