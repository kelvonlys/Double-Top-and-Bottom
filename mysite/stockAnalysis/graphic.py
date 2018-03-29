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
    fig=Figure()
    ax=fig.add_subplot(111)
    y=df['Low']
    x=df['Date']
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot_date(x,df['High'], '-')

    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    graphic=django.http.HttpResponse(content_type='image/png')
    
    canvas.print_png(graphic)
    return graphic





