import pandas as pd
import csv
import numpy
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

filePath = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskThree/Task 3 and 4_Loan_Data.csv"

#miles_per_week =      [ 37  ,39 , 46 , 51 , 88 , 17 , 18 , 20 , 21 , 22 , 23 , 24 , 25 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 38 , 40 , 42 , 57 , 68 , 35  , 36  , 41  , 43  , 45 , 47  , 49  , 50  , 52 , 53  , 54  , 55  , 56  , 58  , 59  , 60  , 61  , 63 , 64  , 65  , 66  , 69  , 70  , 72  , 73  , 75 , 76  , 77  , 78  , 80  , 81  , 82  , 83  , 84 , 85  , 86  , 87  , 89  , 91  , 92  , 93  , 95 , 96  , 97  , 98  , 99  , 100 , 101 , 102 , 103 , 104 , 105 , 106 , 107 , 109 , 110 , 111 , 113 , 114 , 115 , 116 , 116 , 118 , 119 , 120 , 121 , 123 , 124 , 126 , 62  , 67  , 74  , 79  , 90  , 112  ]
#completed_50m_ultra = ['no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','yes','yes','yes','yes','no','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes',]

cols = ["customer_id", "credit_lines_outstanding", "loan_amt_outstanding",
        "total_debt_outstanding", "income", "years_employed", "fico_score"]

customer_id = []
credit_lines_outstanding = []
loan_amt_outstanding = []
total_debt_outstanding = []
income = []
years_employed = []
fico_score = []
default = []

results = []

scaler = StandardScaler()

#pd.DataFrame(confusion_matrix(y_test, y_pred)), columns=column).to_csv("C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskThree/results.csv", index=False)

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

print(len(default))

d = {'credit_lines_outstanding' : credit_lines_outstanding,
    'loan_amt_outstanding': loan_amt_outstanding,
    'total_debt_outstanding': total_debt_outstanding,
    'income': income,
    'years_employed': years_employed,
    'fico_score': fico_score,
    'default': default}

#numpy.array(customer_id, dtype=float)
#numpy.array(credit_lines_outstanding, dtype=float)
#numpy.array(loan_amt_outstanding, dtype=float)
#numpy.array(total_debt_outstanding, dtype=float)
#numpy.array(income, dtype=float)
#numpy.array(years_employed, dtype=float)
#numpy.array(fico_score, dtype=float)
#numpy.array(default, dtype=float)


df = pd.DataFrame(data=d).astype(float)

finished_race = ['no', 'yes']
enc = OrdinalEncoder(categories= [finished_race])
#df['default'] = enc.fit_transform(df[['default']])

#plt.scatter(df.variable, df.default)

sns.countplot(x='default', data=df)
    #plt.show()

x = df.iloc[:,0:6]
y = df['default']
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8, random_state=11)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_test ,y_test)
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))

#pd.DataFrame(results, columns=cols).to_csv("C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskThree/results.csv", index=False)
#reports = []

#for variable_name, variable in [
 #   ("credit_lines_outstanding", credit_lines_outstanding),
  #  #("loan_amt_outstanding", loan_amt_outstanding),
   # ("total_debt_outstanding", total_debt_outstanding),
    #("income", income),
    #("years_employed", years_employed),
    #("fico_score", fico_score)
#]:
 #   y_test, y_pred = logisticRegression(variable, default)

  #  report = classification_report(
   #     y_test,
    #    y_pred,
     #   output_dict=True
    #)

    #report_df = pd.DataFrame(report).transpose()

    # Add a column so we know which variable produced this report
    #report_df["variable"] = variable_name

    #reports.append(report_df)

# Combine all reports into one DataFrame
#combined_reports = pd.concat(reports)

#combined_reports.to_csv("results.csv")
#print("printed!")
