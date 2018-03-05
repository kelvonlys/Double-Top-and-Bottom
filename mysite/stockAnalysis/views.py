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
from detect_peaks import detect_peaks

close = numpy.random.random(100)


#setup for debug
import logging
logging.basicConfig(level=logging.INFO)

API = "z3xWGdrRdxGC1sFtyimA"

result = ""
buy_price = 0
sell_price = 0
rsiPrice = 0
spread_price = 0
double_top = "No"
double_bottom = "No"



def index(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            global result
            result = form.cleaned_data['stockNum']
            getStockInfo (result)
            #getRSI (result)
            return render(request, 'stockAnalysis/basic.html', {'price': [result, buy_price, sell_price, double_top, double_bottom]})
    else:
        form = NameForm()
    return render(request, 'stockAnalysis/home.html', {'form': form})

def getStockInfo (num): 
    global spread_price
    df = quandl.get("HKEX/"+num, start_date="2018-01-01", end_date="2018-03-04", authtoken=API)
    df_reset = df.reset_index()
    df_reframe = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    df_reframe = df_reframe.dropna(how='any')
    df_reframe = df_reframe.to_records(index=False)
    rsiPrice = df['Previous Close']
    spread_price = find_spread(df_reframe['Previous Close'][0])
    find_pattern(df_reframe, "Low")
    find_pattern(df_reframe, "High")


def find_pattern(obj, x):
    spread_vector = 5
    global buy_price
    global sell_price
    global double_top
    global double_bottom
    price = numpy.array(obj[x])
    if (x == "Low"):
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1, valley=True)
    else:
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1) #you can fine tune the thres to smaller value to get even shorter period
    print("index: ", indexes)
    for i in range (0, indexes.size):
        for j in range (i + 1, indexes.size):
            num1 = obj[indexes[i]][x]
            num2 = obj[indexes[j]][x]
            date1 = obj[indexes[i]]['Date']
            date2 = obj[indexes[j]]['Date']
            volume1 = obj[indexes[i]]['Share Volume (000)']
            volume2 = obj[indexes[j]]['Share Volume (000)']
            #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
            #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
            if (num1-num2)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
            else:
                diff = abs(num1-num2)
            if (x == "Low"):
                if (diff <= spread_price*spread_vector):
                    print("double bottom captured, if you see the next message, volume passed too :D")
                    if  (volume1 > volume2): 
                        #print("double bottom captured, info as below: ")
                        #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                        #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                        #print(spread_price*spread_vector)
                        buy_price = min(num1,num2)
                        double_bottom = "Yes"
                else:
                    for k in range (0, len(obj)):
                        norm_price = obj[k][x]
                        adjusted_price = min(num1,num2)
                        if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                            diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
                        else:
                            diff = abs(adjusted_price-norm_price)
                        if (diff <= spread_price*spread_vector):
                            if  (volume1 > volume2): 
                                buy_price = min(adjusted_price,norm_price)
                                double_top = "Yes"
                            else:
                                buy_price = min(adjusted_price,norm_price)
                                double_top = "Yes but without volume"
                                print("double top found without volume")
                        else:
                            buy_price = min(adjusted_price,norm_price)
            else: # (x == "High")
                if (diff <= spread_price*spread_vector):
                    print("double top captured, if you see the next message, volume passed too :D")
                    if  (volume1 > volume2): 
                        sell_price = max(num1,num2)
                        double_top = "Yes"
                        print("double bottom captured, info as below: ")
                        print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                        print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                else:
                    for k in range (0, len(obj)):
                        norm_price = obj[k][x]
                        adjusted_price = max(num1,num2)
                        if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                            diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
                        else:
                            diff = abs(adjusted_price-norm_price)
                        #print("lowest K: ", num1)
                        if (diff <= spread_price*spread_vector):
                            if  (volume1 > volume2): 
                                sell_price = max(adjusted_price,norm_price)
                                double_top = "Yes"
                            else:
                                sell_price = max(adjusted_price,norm_price)
                                double_top = "Yes but without volume"


        
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
getStockInfo("00700")
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
    return render(request, 'stockAnalysis/basic.html',{'origin':[result, buy_price, sell_price]})





