#use this to convert .dat to .csv
import csv

aus = [i.strip().split(',') for i in open("./australian.dat").readlines()]
with open("./australian.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(aus)
'''

#use it read from csv
from __future__ import print_function 
import pandas as pd

aus = pd.read_csv('australian.csv')
print (aus)
'''
