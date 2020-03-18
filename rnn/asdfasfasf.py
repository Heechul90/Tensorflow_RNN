##### Sensor Anomaly Detection

### Bearing Failure Anomaly Detection
# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# set random seed
seed(10)
set_random_seed(10)


### Data loading and pre-processing
# load, average and merge sensor samples
data_dir = 'dataset/bearing/'
filename = 'day3_data.csv'
merged_data = pd.DataFrame()

dataset = pd.read_csv(data_dir + filename, header=None, sep=',')

dataset.columns = ['Date', 'Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
dataset = dataset.set_index('Date')
dataset.index = pd.to_datetime(dataset.index)
dataset.head()
dataset.info()

# transform data file index to datetime and sort in chronological order
merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()
merged_data.to_csv('Averaged_BearingTest_Dataset.csv')
print("Dataset shape:", merged_data.shape)
merged_data.head()

