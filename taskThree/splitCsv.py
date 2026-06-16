import csv
import pandas as pd

filePath = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskThree/Task 3 and 4_loan_Data.csv" 

customer_id = []
credit_lines_outstanding = []
loan_amt_outstanding = []
total_debt_outstanding = []
income = []
years_employed = []
fico_score = []
default = []
lineCount = 0


with open(filePath, "r") as file:
    #lines = file.readlines()
    #print (lines)10
    # for line in lines[1:]:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for _ in reader:
        lineCount += 1
        print(lineCount)

lineCountTrain = (lineCount//100)* 80
print(lineCountTrain)

cols = ["customer_id", "credit_lines_outstanding", "loan_amt_outstanding",
        "total_debt_outstanding", "income", "years_employed", "fico_score", "default"]

train_rows = []
test_rows = []

with open(filePath, "r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for i, row in enumerate(reader): 
        if i < lineCountTrain:
            train_rows.append(row)
        else:
            test_rows.append(row)

pd.DataFrame(train_rows, columns=cols).to_csv("train.csv", index=False)
pd.DataFrame(test_rows, columns=cols).to_csv("test.csv", index=False)
print("Saved results to csv")    