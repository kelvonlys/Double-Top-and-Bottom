############ for search #############
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect #you should delete this later on
from .forms import NameForm

from datetime import date, timedelta
import datetime
import pandas as pd
import quandl
import json
from math import log10, floor


from fbprophet import Prophet
import matplotlib.pyplot as plt

############# TabLib RSI ##############
import numpy as np
import talib

############# for min/max detection ###########
import peakutils
from detect_peaks import detect_peaks

close = np.random.random(100)


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
df_reframe = []
today = np.datetime64('today','D')

def index(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            global result
            result = form.cleaned_data['stockNum']
            search_stock (result)
            #getRSI (result)
            return render(request, 'stockAnalysis/basic.html', {'price': [result, buy_price, sell_price, double_top, double_bottom, spread_price]})
    else:
        form = NameForm()
    return render(request, 'stockAnalysis/home.html', {'form': form})

def search_stock (num): 
    global spread_price
    global df_reframe
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(10,'W')
    print("start_date: ",start_date)
    print("end_date: ",end_date)
    df = quandl.get("HKEX/"+num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
    df_reset = df.reset_index()
    df_reframe = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    df_reframe = df_reframe.dropna(how='any')
    df_reframe = df_reframe.to_records(index=False)
    rsiPrice = df['Previous Close']
    spread_price = find_spread(df_reframe['Previous Close'][0])
    
    period_vector = 30
    date = df_reframe['Date'][1]
    date2 = date + np.timedelta64(period_vector,'D') # can be 'h', 'D' or 'W' or 'Y'
    print("date2: ", date)
    print("date2: ", date2)
    
    find_pattern(df_reframe, "Low")
    find_pattern(df_reframe, "High")


def get_stock_info ():
    return df_reframe    

def find_pattern(obj, x):
    spread_vector = 15
    global buy_price
    global sell_price
    global double_top
    global double_bottom
    price = np.array(obj[x])
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
                    #print("double bottom captured, if you see the next message, volume passed too :D")
                    if  (volume1 > volume2): 
                        buy_price = min(num1,num2) #this will return the latest lowest prices num1, num2
                        double_bottom = "Yes"
                        #print("double bottom captured, info as below: ")
                        #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                        #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    else:
                        buy_price = min(num1,num2)
                        double_bottom = "Pattern not found, trough but without volume"
                else: #this should be short term: within 3months (should deal with this later!!!)
                    single_bottom(obj,x, num1, num2, volume1, volume2, spread_vector)
            else: # (x == "High")
                if (diff <= spread_price*spread_vector):
                    #print("double top captured, if you see the next message, volume passed too :D")
                    if  (volume1 > volume2): 
                        sell_price = max(num1,num2)
                        double_top = "Yes"
                        print("double top captured, info as below: ")
                        print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                        print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    else:
                        sell_price = max(num1,num2)
                        double_top = "Pattern not found, peak but without volume"
                else:
                    single_top(obj,x, num1, num2, volume1, volume2, spread_vector)

def single_top (obj, x, num1, num2, volume1, volume2, spread_vector):
    print("calling normal_price_pattern")
    global sell_price
    global double_top
    for k in range (0, len(obj)):
        norm_price = obj[k][x]
        adjusted_price = max(num1,num2)
        if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
            diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
        else:
            diff = abs(adjusted_price-norm_price)
        if (diff <= spread_price*spread_vector):
            if  (volume1 > volume2): 
                sell_price = max(adjusted_price,norm_price)
                double_top = "Yes but not peak"
            else:
                sell_price = max(adjusted_price,norm_price)
                double_top = "Yes but not peak and without volume"
        else:
            sell_price = max(adjusted_price, norm_price)
            double_top = "No double top pattern is detected, recommendation made according to the latest highest price"

def single_bottom (obj, x, num1, num2, volume1, volume2, spread_vector):
    global buy_price
    global double_bottom
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
                double_bottom = "Yes but not trough"
            else:
                buy_price = min(adjusted_price,norm_price)
                double_bottom = "Yes but not trough and without volume"
                #print("double top found without volume")
        else:
            buy_price = min(adjusted_price,norm_price)
            double_bottom = "No double bottom pattern is detected, recommendation made according to the latest lowest price"
        
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
dfRename['y'] = np .log(dfRename['y'])
print("df rename: ", dfRename)
model = Prophet(changepoint_prior_scale = 0.05)
model.fit(dfRename)
future = model.make_future_dataframe(periods=366)
forecast = model.predict(future)
print("result: ", model.changepoints)'''

#search_stock("00939")


def getRSI (num):
    df = quandl.get("HKEX/"+num, rows=720, authtoken=API)
    rsiPrice = df['Previous Close'].values
    print("RSI: ",talib.RSI(rsiPrice, timeperiod=14))

def doubleTopBottom(request):
    return render(request, 'stockAnalysis/basic.html',{'origin':[result, buy_price, sell_price]})





