from task1example import build_model


# let's jump into task 2

#somehow import price model from the example answer

# estimate prices
from datetime import datetime
from dateutil.relativedelta import relativedelta

interpolate = build_model(csv_path='natgas_R.csv')

#take the input and withdrawal and return the change 
injectionDate = input("Enter the date you want to buy the gas (MM/DD/YY):")
injectionDate = datetime.strptime(injectionDate, "%m/%d/%y")

extractionDate = input("Enter the date you want to sell the gas (MM/DD/YY):")
extractionDate = datetime.strptime(extractionDate, "%m/%d/%y")

quantity = int(input("Enter the quantity of gas you want to store (in barrels):"))

#injection - extraction (rough estimate)

# Time between injection and extraction
diff = relativedelta(extractionDate, injectionDate)
differenceMonth = diff.months + diff.years * 12

#print(differenceMonth) correct
# the time between buying the gas and selling it

import math #for some reason they like importing modules as they're about to use them

#BELOW DOESN'T MATTER
#(take into consideration delay to get/remove gas from storage)
#fillRate = 1200000 # a day
#delay = math.ceil(quantity/fillRate) #day
#delay = delay * 2 #for removal as well as filling
#delay = 1 #day per 1200000 barrels 
#storageDay = differenceDay + delay

#(take into consideration the maximum that can be stored)
tankSize = 1500000 #barrels
tankCost = 10000  #per month

tanksRequired = math.ceil(quantity/tankSize) 
#print(tanksRequired) correct
# rounds up so there is always a tank
additionalCostPerMonth = tanksRequired * tankCost
#print(additionalCostPerMonth) correct
#additional cost x months used for 

storageCost = differenceMonth * additionalCostPerMonth
#print(storageCost) correct

#subtract storage costs

import pandas as pd

price1 = interpolate(pd.Timestamp(injectionDate))
price2 = interpolate(pd.Timestamp(extractionDate))

print(price2- price1)

profit = ((price2 - price1) - storageCost)

print(profit)