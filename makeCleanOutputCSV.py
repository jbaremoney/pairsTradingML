'''
Cleaning the correlation testing results, corrOutput.txt
'''
import ast
import csv
import re

with open('corrOutput.txt', 'r') as file:
    data = file.read()

cleaned_data = re.sub(r'np\.float64\((.*?)\)', r'\1', data)

results = ast.literal_eval(cleaned_data)

with open('cleanCorrResults.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(['Coin1', 'Coin2', 'Correlation', 'ADF Statistic', 'p-value'])

    for row in results:
        writer.writerow(row)