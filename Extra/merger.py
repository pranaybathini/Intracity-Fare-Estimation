import pandas as pd
import csv
with open('result1.csv') as f1,open('result2.csv') as f2,open('output.csv','w') as f:
	writer = csv.writer(f)
	ps = ['ID','FARE']
	writer.writerow(ps)
	data1 = csv.reader(f1,delimiter=',')
	data2 = csv.reader(f2,delimiter=',')
	for row in data1:
		writer.writerow([row[0],row[1]])
	for row in data2:
		writer.writerow([row[0],row[1]])
	
	
