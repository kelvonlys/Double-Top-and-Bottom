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
double_top = "No"
double_bottom = "No"
df_reframe = []
today = np.datetime64('today','D')
global transaction_fee
transaction_fee = 0.003443939

def index(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            global result
            result = form.cleaned_data['stockNum']
            search_stock (result)
            #getRSI (result)
            return render(request, 'stockAnalysis/basic.html', {'price': [result, buy_price, sell_price, double_top, double_bottom, spread_price, success_rate, bottom_success_rate, double_top_profit_average, double_top_profit , hold_till_profit_average, hold_till_profit, buying_points.size]})
    else:
        form = NameForm()
    return render(request, 'stockAnalysis/home.html', {'form': form})

def search_stock (num): 
    global spread_price
    global df_reframe
    global spread_vector
    global stock_num
    global df
    stock_num = num
    period_vector = 52*2
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(period_vector,'W')
    print("start_date: ",start_date)
    print("end_date: ",end_date)
    print("stock number: ", stock_num)
    #df = quandl.get("HKEX/"+stock_num, start_date="2016-04-15", end_date="2018-04-12", authtoken=API)#HSI:BCIW/_HSI
    df = quandl.get("HKEX/"+stock_num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
    df_reset = df.reset_index()
    df_reframe = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    df_reframe = df_reframe.dropna(how='any')
    df_reframe = df_reframe.to_records(index=False)
    rsiPrice = df['Previous Close']
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
    print("find pattern: ", df_reframe['Low'][0])
    for i in range (0, len(df_reframe)):
        if (df_reframe['Low'][i] < 0.5):
            thres = 0
            break
        else:
            thres = 0.02/max(price)
    print("threshold: ", 0.02/max(price))
    if (x == "Low"):
        indexes = detect_peaks(price, threshold=thres, mpd=1, valley=True)
        bottom_pattern(obj, x, indexes)
    else:
        indexes = detect_peaks(price, threshold=thres, mpd=1) #you can fine tune the thres to smaller value to get even shorter period
        top_pattern(obj, x, indexes)
def top_pattern(obj, x, indexes):
    global max_pairs
    global spread_price
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
                #diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
                diff = round(abs(num1-num2), 5)
            else:
                diff = abs(num1-num2)
            spread_price = find_spread(num1)*find_vector(num1)
            if (diff <= find_spread(num1)*find_vector(num1)):
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
                if(obj[j]['High'] > (max_pairs[i]['Price'][1]+find_spread(max_pairs[i]['Price'][1])*find_vector(max_pairs[i]['Price'][1]))):
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
                #diff = round(abs(num1-num2), -int(floor(log10(abs(num1-num2)))))
                diff = round(abs(num1-num2), 5)
            else:
                diff = abs(num1-num2)
            if (diff <= find_spread(num1)*find_vector(num1)):
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
    print("min_pairs", min_pairs)

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
                if(obj[j]['Low'] < (min_pairs[i]['Price'][1]-find_spread(min_pairs[i]['Price'][1])*find_vector(min_pairs[i]['Price'][1]))):
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


def cal_double_top():
    global double_top_profit_average
    global double_top_profit
    double_top_profit_average = 0
    double_top_profit = 0
    double_top_sum = 0
    temp_buying_points = np.copy(buying_points)

    ##### for selling if price breaks the double bottom ####### safest but with less profit
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
                    total_fee = transaction_fee*(sell_price + temp_buying_points[i]['Price'])
                    double_top_profit = double_top_profit + (selling_price - temp_buying_points[i]['Price'] - total_fee)*100/temp_buying_points[i]['Price']
                    #double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price'] - total_fee
                    #print("sell_price: ", selling_price, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
                    #double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price']
                    #print("double_top_profit: ", double_top_profit)
                    #print("double_top_sum: ", double_top_sum)
                    break
            elif (j == max_pairs.size-1):
                selling_price = df_reframe[df_reframe.size-1]['Low']
                total_fee = transaction_fee*(sell_price + temp_buying_points[i]['Price'])
                double_top_profit = double_top_profit + (selling_price - temp_buying_points[i]['Price'] - total_fee)*100/temp_buying_points[i]['Price']
                #double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price'] - total_fee
                #double_top_profit = double_top_profit + selling_price - temp_buying_points[i]['Price'] 
                #print("sell_price: ", selling_price)
                #print("double_top_profit recently: ", double_top_profit, "date: ", transaction_date_adder(max_pairs[j]['Date'][1], 1)['Date'])
                #print("double_top_sum: ",double_top_sum)
    if (temp_buying_points.size > 0):
        double_top_profit_average = double_top_profit/temp_buying_points.size
    print("double_top_profit %: ", double_top_profit_average)
    print("sum: ", double_top_profit)

def cal_hold_till():
    global hold_till_profit_average
    global hold_till_profit
    hold_till_profit_average = 0
    hold_till_profit = 0
    hold_till_sum = 0
    for i in range (0, buying_points.size):
        #selling_price = transaction_date_adder(buying_points[i]['Date'], 13)['High']
        selling_price = df_reframe[df_reframe.size-1]['Low']
        total_fee = transaction_fee*(sell_price + buying_points[i]['Price'])
        #hold_till_profit = hold_till_profit + (selling_price - buying_points[i]['Price']) - total_fee
        hold_till_profit = hold_till_profit + (selling_price - buying_points[i]['Price'] - total_fee)*100/buying_points[i]['Price']
        #print("hold_till_profit: ", hold_till_profit, "date: ", buying_points[i]['Date'])
        #print("hold_till_sum: ", hold_till_sum)
    if (buying_points.size > 0):
        hold_till_profit_average = hold_till_profit/buying_points.size
    print("hold_till_profit: ", hold_till_profit_average)
    print("hold-till sum: ", hold_till_profit)


def recommendation(x):  #reverse recommendation would be faster
    #this should only calculate things within three months
    global buy_price
    global sell_price
    global double_top
    global double_bottom
    global recom_bottom
    global loop_breaker

    loop_breaker = False

    recom_bottom = np.array([],dtype=[('Price', object), ('Date', object), ('Type', object)])
    recommended_period = 52
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(recommended_period,'W')
    #print("start_date : ",start_date)
    #print("end_date: ",end_date)
    #df = quandl.get("HKEX/"+stock_num, start_date="2018-01-18", end_date="2018-04-12", authtoken=API)#HSI:BCIW/_HSI
    df = quandl.get("HKEX/"+stock_num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
    df_reset = df.reset_index()
    obj = pd.DataFrame(df_reset, columns=['Date', 'High','Low', 'Share Volume (000)','Previous Close'])
    obj = obj.dropna(how='any')
    obj = obj.to_records(index=False)
    price = np.array(obj[x])
    new_indexes = np.array([])

    if (x == "Low"):
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1, valley=True)
    else:
        indexes = detect_peaks(price, threshold=0.02/max(price), mpd=1) #you can fine tune the thres to smaller value to get more peaks

    for i in reversed (indexes):
        new_indexes = np.append(new_indexes, int(i))

    
    for i in range (0, new_indexes.size): ##### the array is reversed here, started from the most recent date
        indexes1 = int(new_indexes[i])
        for j in range (i + 1, new_indexes.size):
            indexes2 = int(new_indexes[j])
            reverse_num1 = obj[indexes1][x]
            reverse_num2 = obj[indexes2][x]
            volume1 = obj[indexes1]['Share Volume (000)']
            volume2 = obj[indexes2]['Share Volume (000)']
            date1 = obj[indexes1]['Date']
            date2 = obj[indexes2]['Date']
            print("High/Low? ", x)
            print("i: ", reverse_num1, " i's volume: ", volume1, "date: ", date1)
            print("j: ", reverse_num2, 'i volume: ', volume2, "date: ", date2)
            if (reverse_num1 - reverse_num2)!= 0: #to handle the case log10(0) which would result in math error 
                #diff = round(abs(reverse_num1-reverse_num2), -int(floor(log10(abs(reverse_num1-reverse_num2)))))
                diff = round(abs(reverse_num1-reverse_num2), 5)
            else:
                diff = abs(reverse_num1-reverse_num2)
            if (x == "Low"):
                if (diff <= find_spread(reverse_num1)*find_vector(reverse_num1)):
                    if  (volume1 < volume2): 
                        temp = min(reverse_num1, reverse_num2) ### more checking here
                        double_bottom = "Yes"
                        print("buy_price: ",temp)
                        print("df_reframe: ",df_reframe[df_reframe.size-1]['Low'] )
                        if (temp <= df_reframe[df_reframe.size-1]['Low']):
                            buy_price = temp
                            print("low", df_reframe[df_reframe.size-1]['Low'])
                            loop_breaker = True
                            break
                    else:
                        temp = min(reverse_num1, reverse_num2)
                        double_bottom = "Double trough but without volume"
                        if (temp <= df_reframe[df_reframe.size-1]['Low']):
                            buy_price = temp
                            loop_breaker = True
                            break
                elif(diff > find_spread(reverse_num1)*find_vector(reverse_num1)): #this should be short term: within 3months (should deal with this later!!!)
                    single_bottom(obj, new_indexes)
                    if (loop_breaker == True):
                        break
                
                elif (reverse_num1 > reverse_num2):
                    break
                
            else: # (x == "High")
                if (diff <= find_spread(reverse_num1)*find_vector(reverse_num1)):
                    if  (volume1 < volume2): 
                        temp_sell = max(reverse_num1,reverse_num2)
                        double_top = "Yes"
                        print("double top: Yes")
                        if (temp_sell >= df_reframe[df_reframe.size-1]['High']):
                            sell_price = temp_sell
                            loop_breaker = True
                            break
                    else:
                        temp_sell = max(reverse_num1,reverse_num2)
                        double_top = "Double peak but without volume"
                        if (temp_sell >= df_reframe[df_reframe.size-1]['High']):
                            sell_price = temp_sell
                            loop_breaker = True
                            break
                elif (diff > find_spread(reverse_num1)*find_vector(reverse_num1)):
                    single_top(obj, new_indexes)
                    if (loop_breaker == True):
                        break

                elif (reverse_num1 < reverse_num2):
                    break
            print("loop_breaker outside: ", loop_breaker)
        if (loop_breaker == True):
            break
    
    if (min_pairs.size > 0 and buy_price == min_pairs[min_pairs.size-1]['Price'][1]):
        double_bottom = "Yes"
    if (max_pairs.size > 0 and sell_price == max_pairs[max_pairs.size-1]['Price'][1]):
        double_top = "Yes"

def single_top (obj, indexes):
    global sell_price
    global double_top 
    global loop_breaker

    for i in range (0, indexes.size):
        indexes1 = int(indexes[i])
        adjusted_price = obj[indexes1]['High']
        volume1 = obj[indexes1]['Share Volume (000)']
        for k in range (0, len(obj)):
            if (obj[k]['Date'] < obj[indexes1]['Date']): #if date does not match, go to next loop , loop++
                continue

            '''    ###### this is to make sure the recent double top only compares the latest high, this function should stop after two peaks have been looped, test the stock 00083 (date range same as fyp report), also the break should only break this function and the double top pattern should keep on looping
            if (obj[k]['Date'] < obj[int(indexes[0])]['Date']): 
                loop_breaker = True
                break'''

            norm_price = obj[k]['High']
            volume2 = obj[k]['Share Volume (000)']
            if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(adjusted_price-norm_price), 5)
            else:
                diff = abs(adjusted_price-norm_price)

            if (diff <= find_spread(adjusted_price)*find_vector(adjusted_price)):
                if  (volume1 > volume2): 
                    temp_sell = max(adjusted_price,norm_price)
                    double_top = "Yes, it is a recent double top"
                    #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes1]['Date'], "norm_price: ", norm_price, "Norm date: ",obj[k]['Date'])
                    #print("recent double top: Yes")
                    if (temp_sell >= df_reframe[df_reframe.size-1]['High']):
                        sell_price = temp_sell2
                        loop_breaker = True
                        break
                else:
                    temp_sell = max(adjusted_price,norm_price)
                    double_top = "Recent double top but without volume"
                    #print("loop break: ", loop_breaker)
                    if (temp_sell >= df_reframe[df_reframe.size-1]['High']):
                        sell_price = temp_sell
                        loop_breaker = True
                        break
            else:
                temp_sell = max(adjusted_price, norm_price)
                double_top = "No double top pattern is detected, recommendation made according to the peak detected"
                if (temp_sell >= df_reframe[df_reframe.size-1]['High']):
                    sell_price = temp_sell
                    loop_breaker = True
                    break
        if (loop_breaker == True):
            break

def single_bottom (obj, indexes):
    global buy_price
    global double_bottom
    global loop_breaker
    for i in range (0, indexes.size):
        indexes1 = int(indexes[i])
        adjusted_price = obj[indexes1]['Low']
        volume1 = obj[indexes1]['Share Volume (000)']
        for k in range (0, len(obj)):
            if (obj[k]['Date'] < obj[indexes1]['Date']): #if date does not match, go to next loop , loop++
                continue

            '''    ###### this is to make sure the recent double top only compares the latest high, this function should stop after two peaks have been looped, test the stock 00083 (date range same as fyp report), also the break should only break this function and the double top pattern should keep on looping
            if (obj[k]['Date'] < obj[int(indexes[0])]['Date']): 
                loop_breaker = True
                break'''

            norm_price = obj[k]['Low']
            volume2 = obj[k]['Share Volume (000)']
            if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(adjusted_price-norm_price), 5)
            else:
                diff = abs(adjusted_price-norm_price)

            if (diff <= find_spread(adjusted_price)*find_vector(adjusted_price)):
                if  (volume1 > volume2): 
                    temp_buy = min(adjusted_price,norm_price)
                    double_bottom = "Yes, it is a recent double bottom"
                    #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes1]['Date'], "norm_price: ", norm_price, "Norm date: ",obj[k]['Date'])
                    #print("recent double bottom: Yes")
                    if (temp_buy <= df_reframe[df_reframe.size-1]['Low']):
                        buy_price = temp_buy
                        loop_breaker = True
                        break
                else:
                    temp_buy = min(adjusted_price,norm_price)
                    double_bottom = "Recent double bottom but without volume"
                    print("loop break: ", loop_breaker)
                    if (temp_buy <= df_reframe[df_reframe.size-1]['Low']):
                        buy_price = temp_buy
                        loop_breaker = True
                        print("loop break: ", loop_breaker)
                        break
            else:
                temp_buy = min(adjusted_price, norm_price)
                double_bottom = "No double bottom pattern is detected, recommendation made according to the trough detected"
                if (temp_buy <= df_reframe[df_reframe.size-1]['Low']):
                    buy_price = temp_buy
                    loop_breaker = True
                    break
        if (loop_breaker == True):
            break

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

def find_vector(self):
    if (0.01 <= self <= 0.25):
        return float(2)
    elif  (0.25 < self <= 0.5):
        return float(1)
    elif (0.50 < self <= 1.00):
        return float(2/10)
    elif (1.00 < self <= 2.00):
        return float(2)
    elif (2.00 < self <= 5.00):
        return float(3)
    elif (5.00 < self <= 10.00):
        return float(5)
    elif (10.00 < self <= 20.00):
        return float(10)
    elif (20.00 < self <= 50.00):
        return float(8)
    elif (50.00 < self <= 70.00):
        return float(10)
    elif (70.00 < self <= 100.00):
        return float(20)
    elif (100.00 < self <= 200.00):
        return float(20)
    elif (200.00 < self <= 500.00):
        return float(20)
    elif (1000.00 < self <= 2000.00):
        return float(10)
    elif (2000.00 < self <= 5000.00):
        return float(220)
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

#search_stock("00354")


def getRSI (num):
    df = quandl.get("HKEX/"+num, rows=720, authtoken=API)
    rsiPrice = df['Previous Close'].values
    print("RSI: ",talib.RSI(rsiPrice, timeperiod=14))

def doubleTopBottom(request):
    return render(request, 'stockAnalysis/basic.html',{'origin':[result, buy_price, sell_price]})


'''
def recommendation(x):  #reverse recommendation would be faster
    #this should only calculate things within three months
    global buy_price
    global sell_price
    global double_top
    global double_bottom
    global recom_bottom

    recom_bottom = np.array([],dtype=[('Price', object), ('Date', object), ('Type', object)])
    recommended_period = 12
    end_date = today-np.timedelta64(1,'D')
    start_date = today - np.timedelta64(recommended_period,'W')
    #print("start_date : ",start_date)
    #print("end_date: ",end_date)
    df = quandl.get("HKEX/"+stock_num, start_date="2018-01-18", end_date="2018-04-12", authtoken=API)#HSI:BCIW/_HSI
    #df = quandl.get("HKEX/"+stock_num, start_date=start_date, end_date=end_date, authtoken=API)#HSI:BCIW/_HSI
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
                        double_bottom = "Yes"
                        temp = np.array([((num1, num2), (date1, date2), ("doub"))],dtype=[('Price', object), ('Date', object), ('Type', object)])
                        recom_bottom = np.concatenate((recom_bottom, temp))
                    else:
                        buy_price = min(num1,num2)
                        temp = np.array([((num1, num2), (date1, date2), ("doub_nv"))],dtype=[('Price', object), ('Date', object), ('Type', object)])
                        recom_bottom = np.concatenate((recom_bottom, temp))
                        double_bottom = "Double trough but without volume"
                elif (num2<num1):
                    break
                else: #this should be short term: within 3months (should deal with this later!!!)
                    single_bottom(obj, indexes,date1, date2)
            else: # (x == "High")
                if (diff <= spread_price*spread_vector):
                    if  (volume1 > volume2): 
                        sell_price = max(num1,num2)
                        double_top = "Yes"
                    else:
                        sell_price = max(num1,num2)
                        double_top = "Double peak but without volume"
                elif (num2>num1):
                    break
                else:
                    single_top(obj, indexes)

    for i in reversed (recom_bottom):
        if (i['Type'] == "doub_rec"):
            print("type: ", i['Type'])
            print("pairs: ", i)
            double_bottom = "Yes, it is a recent double bottom"
            buy_price = min(i['Price'][0],i['Price'][0])
            break
        elif (i['Type'] == "doub"):
            print("type: ", i['Type'])
            print("pairs: ", i)
            double_bottom = "Yes"
            buy_price = min(i['Price'][0],i['Price'][0]) #this will return the latest lowest prices num1, num2
            break
            

def single_top (obj, indexes):
    global sell_price
    global double_top 
    

    for i in range (0, indexes.size):
        adjusted_price = obj[indexes[i]]['High']
        volume1 = obj[indexes[i]]['Share Volume (000)']
        #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes[i]]['Date'])
        for k in range (0, len(obj)):
            if (obj[k]['Date'] < obj[indexes[i]]['Date']):
                continue
            norm_price = obj[k]['High']
            volume2 = obj[k]['Share Volume (000)']
            if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
            else:
                diff = abs(adjusted_price-norm_price)

            if (diff <= spread_price*spread_vector):
                if  (volume1 > volume2): 
                    sell_price = max(adjusted_price,norm_price)
                    double_top = "Yes, it is a recent double top"
                else:
                    sell_price = max(adjusted_price,norm_price)
                    double_top = "Yes but not peak and without volume"
                    
            else:
                sell_price = max(adjusted_price, norm_price)
                double_top = "No double top pattern is detected, recommendation made according to the peak detected"

def single_bottom (obj, indexes, date1, date2):
    global buy_price
    global double_bottom
    global recom_bottom
    print("date1 : ",date1, "date2: ", date2)
    for i in range (0, indexes.size):
        adjusted_price = obj[indexes[i]]['Low']
        volume1 = obj[indexes[i]]['Share Volume (000)']
        for k in range (0, len(obj)):
            if (obj[k]['Date'] < obj[indexes[i]]['Date']):
                continue
            norm_price = obj[k]['Low']
            volume2 = obj[k]['Share Volume (000)']
            if (adjusted_price-norm_price)!= 0: #to handle the case log10(0) which would result in math error 
                diff = round(abs(adjusted_price-norm_price), -int(floor(log10(abs(adjusted_price-norm_price)))))
            else:
                diff = abs(adjusted_price-norm_price)
            
            if (diff <= spread_price*spread_vector):
                if  (volume1 > volume2): 
                    buy_price = min(adjusted_price,norm_price)
                    double_bottom = "Yes, it is a recent double bottom"
                    temp = np.array([((adjusted_price, norm_price), (obj[indexes[i]]['Date'], obj[k]['Date']), ("doub_rec"))],dtype=[('Price', object), ('Date', object), ('Type', object)])
                    recom_bottom = np.concatenate((recom_bottom, temp))
                else:
                    buy_price = min(adjusted_price,norm_price)
                    double_bottom = "Yes but not trough and without volume"
                    temp = np.array([((adjusted_price, norm_price), (obj[indexes[i]]['Date'], obj[k]['Date']), ("doub_rec_nv"))],dtype=[('Price', object), ('Date', object), ('Type', object)])
                    recom_bottom = np.concatenate((recom_bottom, temp))
                    #print("double top found without volume")
            #elif (norm_price<adjusted_price):
                    #break
            else:
                buy_price = min(adjusted_price,norm_price)
                #print("num1: ", adjusted_price, " i's volume: ", volume1, "date: ", obj[indexes[i]]['Date'])
                #print("num2: ", norm_price, 'i volume: ', volume2)
                double_bottom = "No double bottom pattern is detected, recommendation made according to the trough detected"
'''



