from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect #you should delete this later on

from .import views

import matplotlib.pyplot as plt
import random
import django
import datetime
import matplotlib.cm as cm
import numpy as np
    
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter


#setup for debug
# interactive chart  -- > https://plot.ly/python/line-charts/
import logging
logging.basicConfig(level=logging.INFO)

API = "z3xWGdrRdxGC1sFtyimA"

def graphic(request):

    df = views.get_stock_info()
    max_pairs = views.get_max_pairs()
    min_pairs = views.get_min_pairs()
    buy_price = views.get_buy_price()
    sell_price = views.get_sell_price()

    fig=Figure(facecolor='#f6f6f6')
    ax=fig.add_subplot(111)
    x=df['Date']
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    
    ax.plot_date(x, df['Low'], '-', color = '#BF463F')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot_date(x, df['High'], '-', color = '#1971E5')
    
    ax.axhline(y=buy_price, color='k')
    ax.axhline(y=sell_price, color='k')

    for i in range (0, max_pairs.size):
        ax.scatter(x=max_pairs[i]['Date'][0], y=max_pairs[i]['Price'][0], marker= 'x', color='r')
        ax.scatter(x=max_pairs[i]['Date'][1], y=max_pairs[i]['Price'][1], marker= 'x', color='r')

    for j in range (0, min_pairs.size):
        #ax.axvline(x=min_pairs[j]['Date'][0], color='#0F892B', linewidth=0.5)
        #ax.axvline(x=min_pairs[j]['Date'][1], color='#D4CD06')

        ax.scatter(x = min_pairs[j]['Date'][0], y=min_pairs[j]['Price'][0], marker= 'x', color='r')
        ax.scatter(x = min_pairs[j]['Date'][1], y=min_pairs[j]['Price'][1], marker= 'x', color='r')

    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    graphic=django.http.HttpResponse(content_type='image/png')
    
    canvas.print_png(graphic)
    return graphic





