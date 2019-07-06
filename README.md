
# What is Pair Trading?


## Agenda



1.   Basic Overview of Stock Market & Securities
2.   Basics of Pair Trading
3.   Illustrating cointegration and spread with fake securities
4.   Testing on historical data
5.   Trading signals and backtesting
6.   Areas of Improvement and Further Steps



## What is gist of stock market trading?

Why do people trade securities on the stock market? In general, securities are simply tradeable financial assets, and in this context, that means shares of stock, or stocks. Owning a share of stock simply means you own a piece of a business/company, and the value of that piece of the pie can vary tremendously, which is where stock price volatility comes in. Due to supply and demand, the more people buy a stock, the higher its price, and the more people sell it, the lower the price. Some people believe that stock prices are an indicator of the actual value of the company, while some believe that it's completely driven by crowd psychology and behavioral factors.

In order to make profits off of trading on the stock market, one needs to buy stock when they anticipate that the price will rise, and then sell the stock when the price rises (ideally before it starts going down). Of course, if it were that simple, we'd all be swimming in money. Many times the prices of various stocks are directly affected by how well the economy is doing. It makes sense, doesn't it? If people feel that a recession is coming soon, people are going to get rid of their shares quickly, and in doing so creating a self-fulfilling prophecy because the value of many stocks will tank.

Some people conduct rigorous and thorough analysis, while others just blindly hope for a turn of good fortune. Either way, this simple type of stock trading is very risky and difficult to master.



## What is Pair Trading?



## Importing Libraries



```
import os
from google.colab import drive
drive.mount('./gdrive')
os.chdir('./gdrive/My Drive/Google Colaboratory/Colab Notebooks/Finance Stuff/Pairs Trading')
```


```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


```

## Generating Fake Securities


```
np.random.seed(107) # So that you can get the same random numbers as me

```

Let's keep this simple and create two fake securities with returns drawn from a normal distribution and created with a random walk


```
# Generate daily returns

Xreturns = np.random.normal(0, 1, 100)

# sum up and shift the prices up

X = pd.Series(np.cumsum(
    Xreturns), name='X') + 50
X.plot(figsize=(15,7))
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_11_0.png)


For the sake of the illustration and intuition, we will generate Y to have a clear link with X, so the price of Y should very in a similar way to X. What we can do is just take X and shift it up slightly and add some noise from a normal distribution.


```
noise = np.random.normal(0, 1, 100)
Y = X + 5 + noise
Y.name = 'Y'

pd.concat([X, Y], axis=1).plot(figsize=(15, 7))

plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_13_0.png)


## Illustrating Cointegration

Let's get this right off the bat. Cointegration is NOT the same thing as correlation! Correlation means that the two variables are interdependent. If you studied statistics, you'll know that correlation is simply the covariance of the two variables normalized by their standard deviations.

Cointegration is slightly different. It means that the ratio between two series will vary around a mean. So a linear combination like:

*Y = αX + e*

would be a stationary time series.

Now what is a stationary time series? In simple terms, it's when a time series varies around a mean and the variance also varies around a mean. What matters most to US is that we know that if a series looks like its diverging and getting really high or low, we know that it will eventually revert back.

I hope I haven't confused you too much. 






```
(Y/X).plot(figsize=(15,7))

plt.axhline((Y/X).mean(), color='red', linestyle='--')

plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_15_0.png)


Here is a plot of the ratio between the two two series. Notice how it tends to revert back to the mean? This is a clear sign of cointegration.

## Cointegration Test

You now know what it means for two stocks to be cointegrated, but how do we actually quantify and test for cointegration?

The module statsmodels has a good cointegration test that outputs a t-score and a p-value. It's a lot of statistical mumbo-jumbo that shows us the probability that we get a certain value given the distribution. In the end, we want to see a low p-value, ideally less than 5%, to give us a clear indicator that the pair of stocks are very likely to be cointegrated.


```
score, pvalue, _ = coint(X,Y)
print(pvalue)

# Low pvalue means high cointegration!
```

    2.0503418653415035e-16


### Clarification
In case you are a bit on the ropes regarding the difference between correlation and cointegration, let me show you some pictures that will make the distinction between correlation and cointegration clear.


```
ret1 = np.random.normal(1, 1, 100)
ret2 = np.random.normal(2, 1, 100)

s1 = pd.Series(np.cumsum(ret1), name='X_divering')
s2 = pd.Series(np.cumsum(ret2), name='Y_diverging')

pd.concat([s1, s2], axis=1).plot(figsize=(15, 7))
plt.show()

print('Correlation: ' + str(s1.corr(s2)))
score, pvalue, _ = coint(s1, s2)
print('Cointegration test p-value: ' + str(pvalue))
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_20_0.png)


    Correlation: 0.9931343801275687
    Cointegration test p-value: 0.881555767469521


Here is a clear example of two series with high correlation but a high pvalue, indicating that they are not cointegrated at all.

Now let's take a look at the opposite side of the spectrum: two series with low correlation but are very cointegrated.


```
Y2 = pd.Series(np.random.normal(0, 1, 800), name='Y2') + 20
Y3 = Y2.copy()

Y3[0:100] = 30
Y3[100:200] = 10
Y3[200:300] = 30
Y3[300:400] = 10
Y3[400:500] = 30
Y3[500:600] = 10
Y3[600:700] = 30
Y3[700:800] = 10

Y2.plot(figsize=(15,7))
Y3.plot()
plt.ylim([0,40])
plt.show()

# very low correlation
print('Correlation: ' + str(Y2.corr(Y3)))
score, pvalue, _ = coint(Y2, Y3)
print('Cointegration test p-value: ' + str(pvalue))
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_22_0.png)


    Correlation: -0.05417844733916934
    Cointegration test p-value: 0.0


## How to Actually make a Pairs Trade

Now that we've clearly explained the essence of pair trading and the concept of cointegration, it's time to get to the nitty-gritty.

We know that if two time series are cointegrated, they will drift towards and apart from each other around the mean. We can be confident that if the two series start to diverge, they will eventually converge later. 

When the series diverge from one another, we say that the *spread* is high. When they drift back towards each other, we say that the *spread* is low. We need to buy one security and short the other. But which ones? 

Remember the equation we had? 

*Y = αX + e* 

As the ratio (Y/X) moves around the mean α, we watch for when X and Y are far apart, which is when α is either too high or too low. Then, when the ratio of the series moves back toward each other, we make money.

In general, we **long the security that is underperforming** and **short the security that is overperforming.**

In terms of the equation, when α is smaller than usual, that means that Y is underperforming and X is overperforming, so we buy Y and sell X.

When α is larger than usual, we sell Y and buy X. 

## Testing on Historical Data

Now let's find some actual securities that are cointegrated based on historical data.


```
def find_cointegrated_pairs(data):
  n = data.shape[1]
  score_matrix = np.zeros((n, n))
  pvalue_matrix = np.ones((n, n))
  keys = data.keys()
  pairs = [] # We store the stock pairs that are likely to be cointegrated
  for i in range(n):
    for j in range(i+1, n):
      S1 = data[keys[i]]
      S2 = data[keys[j]]
      result = coint(S1, S2)
      score = result[0] # t-score
      pvalue = result[1]
      score_matrix[i,j] = score
      pvalue_matrix[i, j] = pvalue
      if pvalue < 0.02:
        pairs.append((keys[i], keys[j]))
  return score_matrix, pvalue_matrix, pairs
```

Now we will download some historical data from the S&P500. It's important to include the market itself into the data because there is such a thing as a *confounding variable* which is when two stocks are not actually cointegrated with each other but with the market, which can mess up our numbers.

I will be using a Python module called Stocker which I cloned from Will Koehrsen's GitHub. 

For libraries, you will need to also install:

*   quandl
*   fbprophet
*   pytrends
*   pystan






```
os.chdir('/content')
```


```
!git clone 'https://github.com/WillKoehrsen/Data-Analysis.git'
```


```
os.chdir('./Data-Analysis/stocker')
```


```
!pip install -U quandl numpy pandas fbprophet matplotlib pytrends pystan
```


```
import stocker
```


```
from stocker import Stocker
```


```
data = pd.DataFrame()
```


```
stocks = ['AAPL', 'ADBE', 'SYMC', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM']
```


```
apple = Stocker('AAPL')
df = apple.make_df('1990-12-12', '2016-12-12')
df = df.set_index(['Date'])
apple_closes = df['Adj. Close']

df.head()
apple_closes.head()
```

    AAPL Stocker Initialized. Data covers 1980-12-12 00:00:00 to 2018-03-27 00:00:00.





    Date
    1990-12-12    1.204414
    1990-12-13    1.238452
    1990-12-14    1.212011
    1990-12-17    1.219609
    1990-12-18    1.284039
    Name: Adj. Close, dtype: float64




```
for ticker in stocks:
  name = str(ticker)
  print(name)
  s = Stocker(name)
  df = s.make_df('2000-12-12', '2016-12-12')
  df = df.set_index(['Date'])
  data[name] = df['Adj. Close']
  
  
```


```
data.head(50)
```


```
from pandas_datareader import data as pdr
import datetime
import fix_yahoo_finance as yf

start_sp = datetime.datetime(2000, 12, 12)
end_sp = datetime.datetime(2016, 12, 12)
yf.pdr_override() 
sp500 = pdr.get_data_yahoo('^GSPC', 
                           start_sp, end_sp)
    
prices = pd.DataFrame()
prices['SP500'] = sp500['Adj Close']
```


```
prices.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SP500</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-12-12</th>
      <td>1371.180054</td>
    </tr>
    <tr>
      <th>2000-12-13</th>
      <td>1359.989990</td>
    </tr>
    <tr>
      <th>2000-12-14</th>
      <td>1340.930054</td>
    </tr>
    <tr>
      <th>2000-12-15</th>
      <td>1312.150024</td>
    </tr>
    <tr>
      <th>2000-12-18</th>
      <td>1322.739990</td>
    </tr>
  </tbody>
</table>
</div>




```
all_prices = pd.merge(prices, data, left_index=True, right_index=True)
```


```
all_prices.head()
stocks = stocks + ['SP500']
```

Now that we've got our data, let's try to find some cointegrated pairs.


```
# Creating a heatmap to show the p-values of the cointegration test

scores, pvalues, pairs = find_cointegrated_pairs(all_prices)
import seaborn
m = [0, 0.2, 0.4, 0.6, 0.8, 1]
seaborn.heatmap(pvalues, xticklabels=stocks,
               yticklabels=stocks, cmap='RdYlGn_r',
               mask = (pvalues >= 0.98))
plt.show()
print(pairs)
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_44_0.png)


    [('ADBE', 'MSFT'), ('SYMC', 'EBAY'), ('JNPR', 'AMD'), ('JNPR', 'IBM')]


According to this heatmap which plots the various p-values for all of the pairs, we've got 4 pairs that appear to be cointegrated. Let's plot their ratios on a graph to see what's going on.


```
S1 = all_prices['ADBE']
S2 = all_prices['MSFT']
score, pvalue, _ = coint(S1, S2)
print(pvalue)
am_ratios = S1 / S2
am_ratios.plot()
plt.axhline(am_ratios.mean())
plt.title('ADBE and MSFT')
plt.legend([' Ratio'])
plt.show()
```

    0.0018312685974441233



![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_46_1.png)



```
S3 = all_prices['SYMC']
S4 = all_prices['EBAY']
score, pvalue, _ = coint(S3, S4)
print(pvalue)
ratios = S3 / S4
ratios.plot()
plt.axhline(ratios.mean())
plt.title('SYMC and EBAY')
plt.legend([' Ratio'])
plt.show()
```

    0.005843043156226274



![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_47_1.png)



```
S5 = all_prices['JNPR']
S6 = all_prices['AMD']
score, pvalue, _ = coint(S5, S6)
print(pvalue)
ratios = S5 / S6
ratios.plot()
plt.axhline(ratios.mean())
plt.title('JNPR and AMD')
plt.legend([' Ratio'])
plt.show()
```

    3.8499134591524897e-16



![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_48_1.png)



```
S7 = all_prices['JNPR']
S8 = all_prices['IBM']
score, pvalue, _ = coint(S7, S8)
print(pvalue)
ratios = S7 / S8
ratios.plot()
plt.axhline(ratios.mean())
plt.title('JNPR and IBM')
plt.legend([' Ratio'])
plt.show()
```

    2.516269559510146e-18



![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_49_1.png)


It appears that our first pair, Adobe and Microsoft, has a plot that moves around the mean in the most stable way. Let's stick with this pair. 

What we need to do next is to try to standardize the ratios because the absolute ratio might not be the most ideal. We need to use z-scores.

Remember from stats class? The z score is calculated by:

###*Z Score (Value) = (Value - Mean) / Standard Deviation*


```
def zscore(series):
  return (series - series.mean()) / np.std(series)
```


```
zscore(am_ratios).plot()
plt.axhline(zscore(am_ratios).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_52_0.png)


By setting two other lines placed at the z-scores of 1 and -1, we can clearly see that for the most part, any big divergences from the mean eventually converge back. This is exactly what we want for pair trading.

## Trading Signals

When conducting any type of trading strategy, it's always important to clearly define and delineate at what point you will actually do a trade. As in, what is the best INDICATOR that I need to buy or sell a particular stock? That's what a trading signal is.

Let's break down a clear plan for creating our trading signals.

### Setup rules

If we're going to look at our ratio and see if it's telling us to buy or sell at a particular moment in time, let's create a prediction variable Y:

*Y = Ratio is buy (1) or sell(-1)*

*Y(t) = Sign(Ratio(t+1) - Ratio(t))*

What's great about pair trading signals is that we don't need to know absolutes about where the prices will go, all we need to know is where it's heading: up or down.

### Train Test Split

When training and testing a model, it's common to have splits like 70/30 or 80/20. Because our data is from 2000-12-12 to 2016-12-12, I'll split it 11 years (~70%) and 5 years (~30%).


```
ratios = all_prices['ADBE'] / all_prices['MSFT']
print(len(ratios))

```

    4024



```
train = ratios[:2017]
test = ratios[2017:]
```

### Feature Engineering

We need to find out what features are actually important in determining the direction of the ratio moves. Knowing that the ratios always eventually revert back to the mean, maybe the moving averages and metrics related to the mean will be important.

Let's try using these features:



*   60 day Moving Average of Ratio
*   5 day Moving Average of Ratio
*   60 day Standard Deviation
*   z score






```
ratios_mavg5 = train.rolling(window=5, center=False).mean()

ratios_mavg60 = train.rolling(window=60, center=False).mean()

std_60 = train.rolling(window=60, center=False).std()

zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
plt.figure(figsize=(15, 7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)

plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_61_0.png)


That's pretty. And enlightening! Let's also take a look at the moving average z-scores.


```
plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_63_0.png)


### Creating a Model

Taking a look at our z-score chart, it's pretty clear that if the absolute value of the z-score gets too high, it tends to revert back. We can keep using our +1/-1 ratios as thresholds, and we can create a model to generate a trading signal:


*   Buy (1) whenever the z-score is below -1.0 because we expect the ratio to increase
*   Sell (-1) whenever the z-score is above 1.0 because we expect the ratio to decrease



### Training and Optimizing

How well does our model work on actual data? Bet you're dying to figure out.


```
plt.figure(figsize=(18,7))

train[160:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5>-1] = 0
sell[zscore_60_5<1] = 0
buy[160:].plot(color='g', linestyle='None', marker='^')
sell[160:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_66_0.png)


So these are the trading signals for the ratios up to around 2009. 

Of course, this is just the trading signals for the ratios, what about the actual stocks?


```
plt.figure(figsize=(18,9))
S1 = all_prices['ADBE'].iloc[:2017]
S2 = all_prices['MSFT'].iloc[:2017]

S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()

# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]

# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

plt.legend(['ADBE', 'MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()
```


![png](images/Nailing%20the%20Basics%20of%20Pairs%20Trading_68_0.png)


**BOOM! How 'bout dat?** That is beautiful. Now we can clearly see when we should buy or sell on the respective stocks.

Let's see how much money we can make off of this strategy, shall we?



```
# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            #print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            #print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            #print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
            
            
    return money
```


```
trade(all_prices['MSFT'].iloc[:2017], all_prices['ADBE'].iloc[:2017], 60, 5)
```




    751.5972913570433



### Backtest on Test Data

Let's test our function on the test data (2010-2016)


```
trade(all_prices['MSFT'].iloc[2018:], all_prices['ADBE'].iloc[2018:], 60, 5)
```




    933.0026201760631



Looks like our strategy is profitable! Given that this data is occuring smack in the middle of the Great Recession, I'd say that's not bad!


## Areas of Improvement and Further Steps

By no means is this a perfect strategy and by no means was the implementation depicted in this article the best. There are several things that can be improved. Feel free to play around with the notebook or python files!

### 1. Using more securities and more varied time ranges

For the pairs trading strategy cointegration test, I only used a handful of stocks. Feel free to test this out on many more, as there are a lot of stocks in the stock market! Also, I only used the time range from 2000 to 2016, which by no means is representative of the average of the stock market in terms of returns or volatility.

### 2. Dealing with overfitting

Anything related to data analysis and training models has much to do with the problem of overfitting, which is simply when a model is trained a bit too closely to the data that it fails to perform when given actual, real data to predict on. There are many different ways to deal with overfitting like validation, Kalman filters, and other statistical methods. 

### 3. Adjusting the trading signals

One thing I noticed about my trading signal algorithm is that it doesn't account for when the stock prices actually overlap and cross each other. Because the code only calls for a buy or sell given their ratio, it doesn't take into account which stock is actually higher or lower. Feel free to improve on this for your own trading strategy!

### 4. More advanced methods

This is just the bare bones of what you can do with algorithmic pair trading. It's super simple because it only deals with moving averages and ratios. If you want to use more complicated statistics, feel free to do so. Some examples of more complex stuff are: the Hurst exponent, half-life of mean reversion, and Kalman filters.


```

```
