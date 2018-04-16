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
            return render(request, 'stockAnalysis/basic.html', {'price': [result, buy_price, sell_price, double_top, double_bottom, spread_price*spread_vector, success_rate, bottom_success_rate, double_top_profit_percentage, hold_till_profit_percentage]})
    else:
        form = NameForm()
    return render(request, 'stockAnalysis/home.html', {'form': form})

def search_stock (num): 
    global spread_price
    global df_reframe
    global spread_vector
    global stock_num
    stock_num = num
    spread_vector = 5
    period_vector = 52*2
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(period_vector,'W')
    print("start_date: ",start_date)
    print("end_date: ",end_date)
    print("stock number: ", stock_num)
    df = quandl.get("HKEX/"+stock_num, start_date="2014-04-15", end_date="2016-04-12", authtoken=API)#HSI:BCIW/_HSI
    #df = quandl.get("HKEX/"+stock_num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
    df_reset = df.reset_index()
    df_reframe = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    df_reframe = df_reframe.dropna(how='any')
    df_reframe = df_reframe.to_records(index=False)
    rsiPrice = df['Previous Close']
    spread_price = find_spread(df_reframe['Previous Close'][0])
    #print("data: ", df_reframe)

    find_pattern(df_reframe, "Low")
    find_pattern(df_reframe, "High")
    recommendation("Low")
    recommendation("High")
    set_buying_point()
    cal_double_top()
    cal_hold_till()



def get_stock_info ():
    return df_reframe    

def get_max_pairs():
    return max_pairs

def get_min_pairs():
    return min_pairs

def get_sell_price():
    return sell_price

def get_buy_price():
    return buy_price

def find_pattern(obj, x):
    price = np.array(obj[x])
    if (x == "Low"):
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1, valley=True)
        bottom_pattern(obj, x, indexes)
    else:
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1) #you can fine tune the thres to smaller value to get even shorter period
        top_pattern(obj, x, indexes)

def top_pattern(obj, x, indexes):
    global max_pairs
    max_pairs = np.array([],dtype=[('Price', object), ('Volume', object), ('Date', object)])
    for i in range (0, indexes.size):
        #print("indexes: ", obj[indexes[i]][x], "date" , obj[indexes[i]]['Date'])
        for j in range (i + 1, indexes.size):
            num1 = obj[indexes[i]][x]
            num2 = obj[indexes[j]][x]
            date1 = obj[indexes[i]]['Date']
            date2 = obj[indexes[j]]['Date']
            volume1 = obj[indexes[i]]['Share Volume (000)']
            volume2 = obj[indexes[j]]['Share Volume (000)']
            if (num1-num2)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
            else:
                diff = abs(num1-num2)
            if (diff <= spread_price*spread_vector):
                if  (volume1 > volume2): 
                    #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                    #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    temp = np.array([((num1, num2),(volume1, volume2), (date1, date2))],dtype=[('Price', object), ('Volume', object), ('Date', object)])
                    max_pairs = np.concatenate((max_pairs, temp))
            elif (num2>num1):
                break
    top_calculation(obj, max_pairs)
    #print("max_pairs", max_pairs)

def top_calculation(obj, max_pairs):
    resistance_period = 1
    success_count = 0
    global success_rate
    success_rate = 0
    for i in range (0, max_pairs.size):
        #print("pairs: ", max_pairs[i]['Date'])
        failure_count = 0
        pair_date = max_pairs[i]['Date'][1]
        for j in range (0, len(obj)):
            if (pair_date <= obj[j]['Date'] <= pair_date + np.timedelta64(resistance_period,'W')):
                #print("date: ", obj[j]['Date'])
                if(obj[j]['High'] > (max_pairs[i]['Price'][1]+spread_price*spread_vector)):
                    next_higher_price = obj[j]['High']
                    #print("next_higher_price: ", next_higher_price)
                    #print("max price: ", max_pairs[i]['Price'][1])
                    #print("date: ", obj[j]['Date'])
                    failure_count += 1
                    #print("count: ", failure_count)
        if (failure_count == 0):
                #print("SUCCESS: max price: ", max_pairs[i]['Price'][1])
                #print("date: ", max_pairs[i]['Date'][1])
                success_count += 1
    print("sucess count: ", success_count)
    if (success_count!=0):
        success_rate = (success_count*100/max_pairs.size)
    print("success_rate: ",success_rate)

def bottom_pattern(obj, x, indexes):
    global min_pairs
    min_pairs = np.array([],dtype=[('Price', object), ('Volume', object), ('Date', object)])
    for i in range (0, indexes.size):
        #print("indexes: ", obj[indexes[i]][x], "date" , obj[indexes[i]]['Date'])
        for j in range (i + 1, indexes.size):
            num1 = obj[indexes[i]][x]
            num2 = obj[indexes[j]][x]
            date1 = obj[indexes[i]]['Date']
            date2 = obj[indexes[j]]['Date']
            volume1 = obj[indexes[i]]['Share Volume (000)']
            volume2 = obj[indexes[j]]['Share Volume (000)']
            if (num1-num2)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
            else:
                diff = abs(num1-num2)
            if (diff <= spread_price*spread_vector):
                if  (volume1 > volume2): 
                    #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                    #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    temp = np.array([((num1, num2),(volume1, volume2), (date1, date2))],dtype=[('Price', object), ('Volume', object), ('Date', object)])
                    min_pairs = np.concatenate((min_pairs, temp))
            elif (num2<num1):
                #print("when num1 > num2, i: ", num1, " i's volume: ", volume1, "date: ", date1)
                #print("when num1 > num2, j: ", num2, 'i volume: ', volume2, "date: ", date2)
                break
    bottom_calculation(obj, min_pairs)
    #print("min_pairs", min_pairs)

def bottom_calculation(obj, min_pairs):
    resistance_period = 1
    success_count = 0
    global bottom_success_rate
    bottom_success_rate = 0
    
    for i in range (0, min_pairs.size):
        #print("pairs: ", min_pairs[i]['Date'])
        failure_count = 0
        pair_date = min_pairs[i]['Date'][1]
        for j in range (0, len(obj)):
            if (pair_date <= obj[j]['Date'] <= pair_date + np.timedelta64(resistance_period,'W')):
                #print("date: ", obj[j]['Date'])
                if(obj[j]['Low'] < (min_pairs[i]['Price'][1]-spread_price*spread_vector)):
                    next_higher_price = obj[j]['Low']
                    #print("next_lower_price: ", next_higher_price)
                    #print("max price: ", min_pairs[i]['Price'][1])
                    #print("lower date: ", obj[j]['Date'])
                    failure_count += 1
                    #print("count: ", failure_count)
        if (failure_count == 0):
                #print("max price: ", min_pairs[i]['Price'][1])
                #print("date: ", min_pairs[i]['Date'][1])
                success_count += 1
    print("bottom sucess count: ", success_count)
    if (success_count!=0):
        bottom_success_rate = (success_count*100/min_pairs.size)
    print("bottom success_rate: ",bottom_success_rate)

def set_buying_point():
    global buying_points
    buying_points = np.array([],dtype=[('Price', object), ('Date', object)])
    for i in range (0, min_pairs.size):
        temp_obj = transaction_date_adder(min_pairs[i]['Date'][1], 1)
        temp = np.array([(temp_obj['Low'], temp_obj['Date'])],dtype=[('Price', object), ('Date', object)])
        buying_points = np.concatenate((buying_points, temp))
    #print("buying_points: ",buying_points) 

def transaction_date_adder(date, period_to_be_added):
    for i in range (0, len(df_reframe)):
        if (date == df_reframe['Date'][i]):
            return df_reframe[i+1]

'''def cal_double_top():
    double_top_profit = 0
    for i in range (0, buying_points.size):
        for j in range (0, max_pairs.size):
            if (max_pairs[j]['Date'][1] >= buying_points[i]['Date']):
                #if (max_pairs[j]['Price'][1] > buying_points[i]['Price']):
                    selling_price = transaction_date_adder(max_pairs[j]['Date'][1], 1)['Low']
                    #print("sell_price: ", selling_price, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
                    double_top_profit = double_top_profit + selling_price - buying_points[i]['Price']
                    #print("double_top_profit: ", double_top_profit)
                    break
            elif (j == max_pairs.size-1):
                selling_price = df_reframe[df_reframe.size-1]['Low']
                double_top_profit = double_top_profit + selling_price - buying_points[i]['Price'] 
                #print("sell_price: ", selling_price)
                #print("double_top_profit recently: ", double_top_profit, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
    print("double_top_profit: ", double_top_profit*100/df_reframe[df_reframe.size-1]['Low'])'''

def cal_double_top():
    global double_top_profit_percentage
    double_top_profit_percentage = 0
    double_top_profit = 0
    temp_buying_points = np.copy(buying_points)
    index_to_del = np.array([],dtype=[('index', object)])

    ##### for selling if price breaks the double bottom ####### but this performance is worse
    '''
    for i in range (0, temp_buying_points.size):
        for k in range (0, len(df_reframe)):
            if (temp_buying_points[i]['Date'] <= df_reframe['Date'][k] <= temp_buying_points[i]['Date'] + np.timedelta64(1,'W')):
                if(df_reframe['Low'][k] < (temp_buying_points[i]['Price']-spread_price*spread_vector)):
                    selling_price = df_reframe['Low'][k]
                    double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price']
                    temp_buying_points[i]['Price'] = 0
                    #print("double_top_profit in test: ", double_top_profit)
                    break
    print("buying_points for i loop: ", buying_points)'''


    for i in range (0, temp_buying_points.size):
        if (temp_buying_points[i]['Price'] == 0):
            continue
        for j in range (0, max_pairs.size):
            if (max_pairs[j]['Date'][1] >= temp_buying_points[i]['Date']):
                #if (max_pairs[j]['Price'][1] > temp_buying_points[i]['Price']):
                    selling_price = transaction_date_adder(max_pairs[j]['Date'][1], 1)['Low']
                    #print("sell_price: ", selling_price, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
                    double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price']
                    #print("double_top_profit: ", double_top_profit)
                    break
            elif (j == max_pairs.size-1):
                selling_price = df_reframe[df_reframe.size-1]['Low']
                double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price'] 
                #print("sell_price: ", selling_price)
                #print("double_top_profit recently: ", double_top_profit, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
    double_top_profit_percentage = double_top_profit*100/df_reframe[df_reframe.size-1]['Low']
    print("double_top_profit %: ", double_top_profit_percentage)

def cal_hold_till():
    global hold_till_profit_percentage
    hold_till_profit_percentage = 0
    hold_till_profit = 0
    for i in range (0, buying_points.size):
        #selling_price = transaction_date_adder(buying_points[i]['Date'], 13)['High']
        selling_price = df_reframe[df_reframe.size-1]['Low']
        hold_till_profit = hold_till_profit + selling_price - buying_points[i]['Price'] 
        #print("hold_till_profit: ", hold_till_profit, "date: ", buying_points[i]['Date'])
    hold_till_profit_percentage = hold_till_profit*100/df_reframe[df_reframe.size-1]['Low']
    print("hold_till_profit: ", hold_till_profit_percentage)
    print("latest price: ", df_reframe[df_reframe.size-1]['Low'])
    print("date: ", df_reframe[df_reframe.size-1]['Date'])


def recommendation(x): 
    #this should only calculate things within three months
    global buy_price
    global sell_price
    global double_top
    global double_bottom

    recommended_period = 12
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(recommended_period,'W')
    #print("start_date : ",start_date)
    #print("end_date: ",end_date)
    df = quandl.get("HKEX/"+stock_num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
    df_reset = df.reset_index()
    obj = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    obj = obj.dropna(how='any')
    obj = obj.to_records(index=False)
    price = np.array(obj[x])

    if (x == "Low"):
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1, valley=True)
    else:
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1) #you can fine tune the thres to smaller value to get even shorter period

    for i in range (0, indexes.size):
        for j in range (i + 1, indexes.size):
            num1 = obj[indexes[i]][x]
            num2 = obj[indexes[j]][x]
            volume1 = obj[indexes[i]]['Share Volume (000)']
            volume2 = obj[indexes[j]]['Share Volume (000)']
            date1 = obj[indexes[i]]['Date']
            date2 = obj[indexes[j]]['Date']
            #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
            #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
            if (num1-num2)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
            else:
                diff = abs(num1-num2)
            if (x == "Low"):
                if (diff <= spread_price*spread_vector):
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
                    single_bottom(obj, indexes)
            else: # (x == "High")
                if (diff <= spread_price*spread_vector):
                    if  (volume1 > volume2): 
                        sell_price = max(num1,num2)
                        double_top = "Yes"
                        #print("double top captured, info as below: ")
                        #print("i: ", num1, " i's volume: ", volume1, "date: ", date1)
                        #print("j: ", num2, 'i volume: ', volume2, "date: ", date2)
                    else:
                        sell_price = max(num1,num2)
                        double_top = "Pattern not found, peak but without volume"
                else:
                    single_top(obj, indexes)

def single_top (obj, indexes):
    global sell_price
    global double_top
    for i in range (0, indexes.size):
        adjusted_price = obj[indexes[i]]['High']
        volume1 = obj[indexes[i]]['Share Volume (000)']
        #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes[i]]['Date'])
        for k in range (0, len(obj)):
            norm_price = obj[k]['High']
            volume2 = obj[k]['Share Volume (000)']
            if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
            else:
                diff = abs(adjusted_price-norm_price)
            if (diff <= spread_price*spread_vector):
                if  (volume1 > volume2): 
                    sell_price = max(adjusted_price,norm_price)
                    double_top = "Yes but not peak"
                    #print("price1: ", norm_price)
                    #print("price2: ", adjusted_price)
                else:
                    sell_price = max(adjusted_price,norm_price)
                    double_top = "Yes but not peak and without volume"
            else:
                sell_price = max(adjusted_price, norm_price)
                #print("num1: ", adjusted_price, " i's volume: ", volume1)
                #print("num2: ", norm_price, 'i volume: ', volume2)
                double_top = "No double top pattern is detected, recommendation made according to the peak detected"

def single_bottom (obj, indexes):
    global buy_price
    global double_bottom
    for i in range (0, indexes.size):
        adjusted_price = obj[indexes[i]]['Low']
        volume1 = obj[indexes[i]]['Share Volume (000)']
        for k in range (0, len(obj)):
            norm_price = obj[k]['Low']
            volume2 = obj[k]['Share Volume (000)']
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
                #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes[i]]['Date'])
                #print("num2: ", norm_price, 'i volume: ', volume2)
                double_bottom = "No double bottom pattern is detected, recommendation made according to the trough detected"


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

#search_stock("00019")


def getRSI (num):
    df = quandl.get("HKEX/"+num, rows=720, authtoken=API)
    rsiPrice = df['Previous Close'].values
    print("RSI: ",talib.RSI(rsiPrice, timeperiod=14))

def doubleTopBottom(request):
    return render(request, 'stockAnalysis/basic.html',{'origin':[result, buy_price, sell_price]})





