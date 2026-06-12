#new project!!
import csv
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from scipy import stats
import pandas as pd

filePath = "C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/input.csv" 
#could've used r and then backslashes but this looks prettier

colour = "g"
dates = []
entries = []
prices = []


with open(filePath, "r") as file:
    #lines = file.readlines()
    #print (lines)
    # for line in lines[1:]:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for line in reader:
        date = (line[0])
        dates.append(date)
        price = (line[1])
        prices.append(float(price))
        print(date)

for entry in range (len(dates)):
    entries.append(entry)
    #print(entries)

#dates = [datetime.strptime(d, "%m/%d/%y") for d in dates]

slope, intercept, r, p, std_err = stats.linregress(entries, prices)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, entries))

speed = myfunc(100)

print(speed)

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.xticks(rotation=45)

plt.scatter(dates, prices, color= colour)
plt.plot(entries, mymodel)
plt.ylabel("Account Value")
plt.xlabel("wager Count")
plt.show()