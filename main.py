import csv

db = []

with open('P2DB.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        db.append(row)


print(db[len(db)-1])