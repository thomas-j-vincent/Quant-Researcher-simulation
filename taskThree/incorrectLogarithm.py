
import csv
import matplotlib.pyplot as plt
import numpy
from sklearn import linear_model
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


with open(filePath, "r") as file:
    #lines = file.readlines()
    #print (lines)10
    # for line in lines[1:]:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for line in reader:
        customer_id.append(line[0]) #nothing to do with default
        credit_lines_outstanding.append(line[1])
        loan_amt_outstanding.append(line[2])
        total_debt_outstanding.append(line[3])
        income.append(line[4])
        years_employed.append(line[5])
        fico_score.append(line[6])
        default.append(line[7])
        #print(date)

logisticRegression(credit_lines_outstanding, default,"credit_lines_outstanding")
logisticRegression(loan_amt_outstanding, default,"loan_amt_outstanding")
logisticRegression(total_debt_outstanding, default,"total_debt_outstanding")
logisticRegression(income, default,"income")
logisticRegression(years_employed, default,"years_employed")
logisticRegression(fico_score, default,"fico_score")

#plt.scatter(dates_numeric, prices, color= colour)
#plt.plot(entries, mymodel)
#plt.plot(dates_numeric, mymodel)
#plt.scatter(inputDate_numeric[0], result[0], color='red', zorder=5, label=f'Prediction: {result[0]:.2f}')
#plt.ylabel("prices")
#plt.xlabel("dates numeric")
#plt.legend()
#plt.show()