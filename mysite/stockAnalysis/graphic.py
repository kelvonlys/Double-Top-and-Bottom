from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect #you should delete this later on

from .import views

import matplotlib.pyplot as plt
import random
import django
import datetime
    
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter

#setup for debug
import logging
logging.basicConfig(level=logging.INFO)

API = "z3xWGdrRdxGC1sFtyimA"

def graphic(request):

    df = views.get_stock_info()
    #max_pairs = views.get_max_pairs()
    min_pairs = views.get_min_pairs()
    fig=Figure()
    ax=fig.add_subplot(111)
    x=df['Date']
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    
    ax.plot_date(x, df['Low'], '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot_date(x,df['High'], '-')

    #for i in range (0, max_pairs.size):
        #ax.axvline(x=max_pairs[i]['Date'][0], color='r')
        #ax.axvline(x=max_pairs[i]['Date'][1], color='r')

    for j in range (0, min_pairs.size):
        ax.axvline(x=min_pairs[j]['Date'][0], color='r')
        ax.axvline(x=min_pairs[j]['Date'][1], color='g')

    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    graphic=django.http.HttpResponse(content_type='image/png')
    
    canvas.print_png(graphic)
    return graphic





