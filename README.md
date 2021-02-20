# Double-Top-and-Bottom
This project applies the commonly used double top and Bottom chart pattern to the HK equity market. A web interface is provided for user to search for particular stock and check whether a double top or double bottom has occurred.

# Data Source
The data used in this project is provided by Qunadl. Quandl API is used to get the historical data of HK stocks in daily resolution. 


# Chart Pattern Detection 
A double top or double bottom is detected when two consecutive peaks/bottoms of approximately the same price on a price-versus-time chart of a market. The first peak/bottom has to have higher volume than the second one as a way of confirmation. 
A chart pattern detection library is used in this project, developed by Marcos Duarte, https://github.com/demotu/BMC. The library returns an array of the crest and trough of a wave which in this case is the tops and bottoms of a stock's price.


# Data Restructuring
Since the data retrieved from Quandl includes lots of extra data which does not fit into the chart detection library, the data has to go through a few steps of reconstruction by utilising the Pandas framework. The data is then packed into a numpy array for pattern detection.

# Success Rate Calculation
Signal is generated when double top or bottom is formed with a confirmation of certain candlestick patterns. This strategy simulates a long or short position right after the signal is generated. Success rate is then calculated by checking if the price breaks the support level in a 5-days period. 

# Result and Visualization
Finally the result is plotted out by using Matplotlib and the success rate is displayed. The crosses are points where double tops and bottoms occur.

![alt text](https://github.com/kelvonlys/Double-Top-and-Bottom/blob/master/00823.png)
