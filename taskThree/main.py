import csv
import matplotlib.pyplot as plt
import numpy
from sklearn import linear_model
from sklearn.preprocessing import OrdinalEncoder 
import pandas as pd

filePathTest = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/Test.csv"
filePathTrain = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/Train.csv"

customer_idTrain = []
credit_lines_outstandingTrain = []
loan_amt_outstandingTrain = []
total_debt_outstandingTrain = []
incomeTrain = []
years_employedTrain = []
fico_scoreTrain = []
defaultTrain = []

def logisticRegression(variable, default, name):

    results = []
    #for i in variable:
    variable = numpy.array(variable, dtype=float).reshape(-1,1)
    default = numpy.array(default)

    logr = linear_model.LogisticRegression()
    
    logr.fit(variable,default)

    def logit2prob(logr, X):
        log_odds = logr.coef_ * X + logr.intercept_
        odds = numpy.exp(log_odds)
        probability = odds / (1 + odds)
        return(probability)

    #predicted = logr.predict(numpy.array([value], dtype=float).reshape(-1,1))
    prob = logit2prob(logr, variable)

    #print("predicted answer:" + str(predicted[0]))
    print("probability:" + str(prob))
    if (numpy.mean(prob) < 0.005):
        print(str(variable) + "not related")
    else: 
        results.append(prob)
        df = pd.DataFrame(results[0])
        df.to_csv(f"{name}.csv", index=False)
        print("Saved results to csv")
    #return(predicted)

#def logisticRegression2():

with open(filePathTrain, "r") as file:
    #lines = file.readlines()
    #print (lines)10
    # for line in lines[1:]:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for line in reader:
        customer_idTrain.append(line[0]) #nothing to do with default
        credit_lines_outstandingTrain.append(line[1])
        loan_amt_outstandingTrain.append(line[2])
        total_debt_outstandingTrain.append(line[3])
        incomeTrain.append(line[4])
        years_employedTrain.append(line[5])
        fico_scoreTrain.append(line[6])
        defaultTrain.append(line[7])

customer_idTrain = []
credit_lines_outstandingTrain = []
loan_amt_outstandingTrain = []
total_debt_outstandingTrain = []
incomeTrain = []
years_employedTrain = []
fico_scoreTrain = []
defaultTrain = []

df = pd.DataFrame(data=d)

default = [0,1]

enc = OrdinalEncoder(categories= [default])
df['defaultTrain']