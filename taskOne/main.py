#new project!!
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from scipy import stats
import pandas as pd

filePath = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskOne/input.csv" 
#could've used r and then backslashes but this looks prettier

colour = "g"
dates = []
entries = []
prices = []
inputDate = input("Enter the date you want to find a price for(MM/DD/YY):")
#inputDate = "10/10/24"


with open(filePath, "r") as file:
    #lines = file.readlines()
    #print (lines)10
    # for line in lines[1:]:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for line in reader:
        date = (line[0])
        dates.append(date)
        price = (line[1])
        prices.append(float(price))
        #print(date)

for entry in range (len(dates)):
    entries.append(entry)
    #print(entries)

dates = [datetime.strptime(d, "%m/%d/%y") for d in dates]
inputDate = [datetime.strptime(inputDate, "%m/%d/%y")]
#print(type(dates))
#print(dates)

dates_numeric = np.array([d.timestamp() for d in dates])
inputDate_numeric = np.array([d.timestamp() for d in inputDate])
#slope, intercept, r, p, std_err = stats.linregress(entries, prices)
slope, intercept, r, p, std_err = stats.linregress(dates_numeric, prices)

def myfunc(x):
  return slope * x + intercept

#mymodel = list(map(myfunc, entries))
mymodel = list(map(myfunc, dates_numeric))

result = myfunc(inputDate_numeric)

print(result)

#ax = plt.gca()

#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

#plt.xticks(rotation=45)

plt.scatter(dates_numeric, prices, color= colour)
#plt.plot(entries, mymodel)
plt.plot(dates_numeric, mymodel)
plt.scatter(inputDate_numeric[0], result[0], color='red', zorder=5, label=f'Prediction: {result[0]:.2f}')
plt.ylabel("prices")
plt.xlabel("dates numeric")
plt.legend()
plt.show()