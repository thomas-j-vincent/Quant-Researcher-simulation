from task1example import build_model

interpolate = build_model(csv_path='natgas_R.csv')
# let's jump into task 2

#somehow import price model from the example answer

# estimate prices
from datetime import datetime

#take the input and withdrawal and return the change 
injectionDate = input("Enter the date you want to buy the gas (MM/DD/YY):")
injectionDate = [datetime.strptime(i, "%m/%d/%y") for i in injectionDate]
extractionDate = input("Enter the date you want to sell the gas (MM/DD/YY):")
extractionDate = [datetime.strptime(i, "%m/%d/%y") for i in extractionDate]
quantity = input("Enter the quantity of gas you want to store (in barrels):")

#injection - extraction (rough estimate)
#find difference
differenceDay = (injectionDate.day - extractionDate.month)
differenceMonth = (injectionDate.month - extractionDate.month)
differenceYear = (injectionDate.year - extractionDate.year)

# if day > 1 month + 1 
if differenceDay > 1:
    differenceMonth += 1

# if month > 12 year + 1
if differenceMonth > 12:
    differenceYear += 1

dateDifference = datetime(differenceMonth, differenceDay, differenceYear)
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
tankCost = 100000  #per month

tanksRequired = math.ceil(quantity/tankSize) 
# rounds up so there is always a tank
additionalCostPerMonth = tanksRequired * tankCost
#additional cost x months used for 

storageCost = differenceMonth * additionalCostPerMonth

#subtract storage costs

import pandas as pd

price1 = interpolate(pd.Timestamp(injectionDate))
price2 = interpolate(pd.Timestamp(extractionDate))

profit = ((price2 - price1) - storageCost)

print(profit)