############ for search #############
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect #you should delete this later on
from .forms import NameForm

import pandas as pd
import quandl
import json

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
    df = quandl.get("HKEX/"+num, returns="numpy", rows=30, authtoken=API)
    priceHigh = df['High']
    priceLow = df['Low']
    nominal = df['Nominal Price']
    rsiPrice = df['Previous Close']
    #calMax(df)
    calMin(df)
    highest = 0
    lowest = min(priceLow)
    

def calMin(obj):
    minPrices = numpy.array(obj['Low'])
    minPrices = 1./minPrices
    indexes = peakutils.indexes(minPrices, thres=0.02/max(minPrices), min_dist=1) #you can fine tune the thres to smaller value to get even shorter period
    print("min: ", indexes)

    for i in range (0, indexes.size):
        print("print i: ",i)
        print("lowest: ", obj[indexes[i]]['Date'])
        print("lowest price: ", obj[indexes[i]]['Low'])
        for j in range (i + 1, indexes.size):
            if obj[indexes[i]]['Low'] == obj[indexes[j]]['Low']:
                print("indexes[i] ", obj[indexes[i]]['Low'])

        


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





