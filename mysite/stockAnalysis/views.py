############ for search #############
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect #you should delete this later on
from .forms import NameForm


import pandas as pd
import quandl
import json
from math import log10, floor


from fbprophet import Prophet
import matplotlib.pyplot as plt

############# TabLib RSI ##############
import numpy
import talib

############# for min/max detection ###########
import peakutils
from scipy import signal

close = numpy.random.random(100)


#setup for debug
import logging
logging.basicConfig(level=logging.INFO)

API = "z3xWGdrRdxGC1sFtyimA"

result = ""
highest = 0
lowest = 0
rsiPrice = 0
spread_price = 0

df = quandl.get("HKEX/"+"00939", returns="numpy", rows=30, authtoken=API)
test = 'Low'
print("testing: ", df[test])

def index(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            global result
            result = form.cleaned_data['stockNum']
            getStockInfo (result)
            #getRSI (result)
            return render(request, 'stockAnalysis/basic.html', {'price': [result, highest, lowest]})
    else:
        form = NameForm()
    return render(request, 'stockAnalysis/home.html', {'form': form})

def getStockInfo (num): 
    global highest 
    global lowest 
    df = quandl.get("HKEX/"+num, returns="numpy", rows=50, authtoken=API)
    priceHigh = df['High']
    priceLow = df['Low']
    nominal = df['Nominal Price']
    rsiPrice = df['Previous Close']
    global spread_price
    spread_price = find_spread(df['Previous Close'][df.size-1])
    #calMax(df)
    calMin(df)
    highest = 0
    lowest = min(priceLow)
    

def calMax(obj):
    maxPrices = numpy.array(obj['High'])
    print("maxPrice: ", maxPrices)
    indexes = peakutils.indexes(maxPrices, thres=0.02/max(maxPrices), min_dist=1) #you can fine tune the thres to smaller value to get even shorter period
    print("max/min: ", indexes)
    x = indexes.size
    print("x: ",x)
    for y in range(indexes.size):
        print("prices:", obj[indexes[y-1]]['High'])
        print("prices: ", obj[indexes[y-1]])

def calMin(obj):
    minPrices = numpy.array(obj['Low'])
    minPrices = 1./minPrices
    indexes = peakutils.indexes(minPrices, thres=0.02/max(minPrices), min_dist=1) #you can fine tune the thres to smaller value to get even shorter period
    print("min: ", indexes)
    for i in range (0, indexes.size):
        #print("lowest: ", obj[indexes[i]]['Date'])
        for j in range (i + 1, indexes.size):
            num1 = obj[indexes[i]]['Low']
            num2 = obj[indexes[j]]['Low']
            date1 = obj[indexes[i]]['Date']
            date2 = obj[indexes[j]]['Date']
            volume1 = obj[indexes[i]]['Share Volume (000)']
            volume2 = obj[indexes[j]]['Share Volume (000)']
            #print("diff: ", round(abs(num1-num2), -int(floor(log10(abs(num1-num2))))))
            diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
            if (diff <= spread_price*10):
                print("double bottom captured, if you see the next message, volume passed too :D")
                if  (volume1 > volume2): 
                    print("double bottom captured, info as below: ")
                    print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                    print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    print(spread_price*10)


        
def find_spread(self):
    #Spread table from HKEX
    if (0.01 <= self <= 0.25):
        return float(0.001)
    elif  (0.25 < self <= 0.5):
        return float(0.005)
    elif (0.50 < self <= 10.00):
        return float(0.010)
    elif (10.00 < self <= 20.00):
        return float(0.020)
    elif (20.00 < self <= 100.00):
        return float(0.050)
    elif (100.00 < self <= 200.00):
        return float(0.100)
    elif (200.00 < self <= 500.00):
        return float(0.200)
    elif (1000.00 < self <= 2000.00):
        return float(1.000)
    elif (2000.00 < self <= 5000.00):
        return float(2.000)
    else:
        return float(5.000)

'''
dfReset = df.reset_index()
dfRename = pd.DataFrame(dfReset, columns=['Date', 'High'])
dfRename.columns = ['ds', 'y']
dfRename['y'] = numpy.log(dfRename['y'])
print("df rename: ", dfRename)
model = Prophet(changepoint_prior_scale = 0.05)
model.fit(dfRename)
future = model.make_future_dataframe(periods=366)
forecast = model.predict(future)
print("result: ", model.changepoints)'''




def getRSI (num):
    df = quandl.get("HKEX/"+num, rows=720, authtoken=API)
    rsiPrice = df['Previous Close'].values
    print("RSI: ",talib.RSI(rsiPrice, timeperiod=14))

def doubleTopBottom(request):
    return render(request, 'stockAnalysis/basic.html',{'origin':[result, highest, lowest]})





