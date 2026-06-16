
import csv
import matplotlib.pyplot as plt

filePath = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskOne/input.csv" 

customer_id = []
credit_lines_outstanding = []
loan_amt_outstanding = []
total_debt_outstanding = []
income = []
years_employed = []
fico_score = []
default = []

def lo

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

plt.scatter(dates_numeric, prices, color= colour)
#plt.plot(entries, mymodel)
plt.plot(dates_numeric, mymodel)
plt.scatter(inputDate_numeric[0], result[0], color='red', zorder=5, label=f'Prediction: {result[0]:.2f}')
plt.ylabel("prices")
plt.xlabel("dates numeric")
plt.legend()
plt.show()