import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 8

# Current Setup for Baseball

def read_in_player(player_id):
  #Read in Player Data
  data = pd.read_csv("""INSERT FILE""")
  
  #Set Index to Game Date
  form = '%b %d %Y'
  times = pd.to_datetime(data['Date'].str.slice(stop=5) + ' ' + data['Year'].astype(int).astype(str), format=form)
  data.set_index(times, inplace=True)
  #DERIVATIVE STAT CREATION
  data.replace([np.inf, -np.inf], 1, inplace=True)
  data.replace(np.nan, 0, inplace=True)
  
  return data
  
def test_stationarity(timeseries, window=162):
  #Determine Rolling Statistics
  rollmean = timeseries.rolling(window).mean()
  rollstd = timeseries.rolling(window).std()
  
  orig = plt.plot(timeseries, color='blue', label='Original')
  mean = plt.plot(rollmean, color='red', label='Rolling Mean')
  std = plt.plot(rollstd, color='black', label='Rolling Std')
  plt.legend(loc='best')
  plt.title('Rolling Mean & Standard Deviation')
  plt.show(block=False)
  
  #Perform Dickey-Fuller test:
  print('Results of Dickey-Fuller Test:')
  dftest = adfuller(timeseries, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations'])
  for key, value in dftest[4].items():
    dfoutput['Critical Value {key}'.format(key=key)] = value
  print(dfoutput)
  
def ACF_PACF_plots(timeseries):
  lag_acf = acf(timeseries, nlags=14)
  lag_pacf = pacf(timeseries, nlags=14, method='ols')
  
  #Plot ACF 
  plt.subplot(121)
  plt.plot(lag_acf)
  plt.axhline(y=0,linestyle='--',color='gray')
  plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
  plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
  plt.title('Autocorrelation Function')
  
  #Plot PACF
  plt.subplot(122)
  plt.plot(lag_pacf)
  plt.axhline(y=0,linestyle='--',color='gray')
  plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
  plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
  plt.title('Partial Autocorrelation Function')
  plt.tight_layout()
  
  return
  
def predict(timeseries):
  model = SimpleExpSmoothing(timeseries)
  model_fit = model.fit()
  yhat = model_fit.predict(len(timeseries), len(timeseries))
  ses_result = yhat
  
  MA_16 = timeseries.rolling(window=16, center=True).mean()
  MA_40 = timeseries.rolling(window=40, center=True).mean()
  MA_81 = timeseries.rolling(window=81, center=True).mean()
  MA_162 = timeseries.rolling(window=162, center=True).mean()
  plt.plot(MA_16, color='yellow', label='MA - Tenth of Season')
  plt.plot(MA_40, color='blue', label='MA - Quarter of Season')
  plt.plot(MA_81, color='red', label='MA - Half Season')
  plt.plot(MA_162, color='black', label='MA - Full Season')
  plt.legend(loc=0)
  print('SES Result: ' + str(ses_result))
  
  return
