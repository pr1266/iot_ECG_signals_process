from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, LSTM, MaxPool1D, Flatten, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, model_to_dot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import mt4_hst

sc = MinMaxScaler(feature_range = (0,1))
sc_ = MinMaxScaler(feature_range = (0,1))
ohe = OneHotEncoder()

#this base model is one branch of the main model
#it takes a time series as an input, performs 1-D convolution, and returns it as an output ready for concatenation
def get_base_model(input_len, fsize):
    #the input is a time series of length n and width 20
    input_seq = Input(shape=(input_len, 20))
    #choose the number of convolution filters
    nb_filters = 10
    #1-D convolution and global max-pooling
    convolved_1 = Conv1D(64, fsize, padding = "same", activation = "tanh")(input_seq)
    max_pool_1 = MaxPool1D()(convolved_1)
    convolved_2 = Conv1D(64, fsize, padding = "same", activation = "tanh")(max_pool_1)
    max_pool_2 = MaxPool1D()(convolved_2)
    flatten_1 = Flatten()(max_pool_2)
    rep_1 = RepeatVector(1)(flatten_1)
    lstm_1 = LSTM(units = 64, return_sequences = True)(rep_1)

    flatten_2 = Flatten()(max_pool_1)
    rep_2 = RepeatVector(1)(flatten_2)
    lstm_2 = LSTM(64, return_sequences = True)(rep_2)

    lstm_3 = LSTM(64, return_sequences = True)(input_seq)
    concat = Concatenate()([lstm_1, lstm_2])

    compressed_1 = Dense(60, activation = "tanh")(concat)
    compressed_1 = Dropout(0.3)(compressed_1)

    
    compressed_2 = Dense(60, activation = "tanh")(compressed_1)
    compressed_2 = Dropout(0.3)(compressed_2)

    model = Model(inputs = input_seq, outputs = compressed_2)
    return model

#this is the main model
#it takes the original time series and its down-sampled versions as an input, and returns the result of classification as an output
def main_model(inputs_lens = [512, 1024, 3480], fsizes = [8,16,24]):
    #the inputs to the branches are the original time series, and its down-sampled versions
    #! inja 3 ta sequence len motafavet darim :
    #! short , mid , long
    input_smallseq = Input(shape=(inputs_lens[0], 20))
    input_medseq = Input(shape=(inputs_lens[1] , 20))
    input_origseq = Input(shape=(inputs_lens[2], 20))
    #the more down-sampled the time series, the shorter the corresponding filter
    base_net_small = get_base_model(inputs_lens[0], fsizes[0])
    base_net_med = get_base_model(inputs_lens[1], fsizes[1])
    base_net_original = get_base_model(inputs_lens[2], fsizes[2])
    embedding_small = base_net_small(input_smallseq)
    embedding_med = base_net_med(input_medseq)
    embedding_original = base_net_original(input_origseq)
    #concatenate all the outputs
    merged = Concatenate()([embedding_small, embedding_med, embedding_original])
    dense_1 = Dense(60, activation = 'sigmoid')(merged)
    dense_2 = Dense(24, activation = 'sigmoid')(dense_1)
    out = Dense(1, activation='sigmoid')(dense_2)
    model = Model(inputs=[input_smallseq, input_medseq, input_origseq], outputs=out)
    return model

def read_data():

    #! 500000 data akharo dar miaram :
    data = mt4_hst.read_hst('crypto_data\EURUSD\EURUSD.hst')
    data = data.drop(['time', 'volume'], axis = 1)
    data = data.iloc[-500000:,:]
    return data

def prepare_data(data, length):
        
    #! 20 000 record ham mizaram vase test :
    data_train = data.iloc[0:len(data) - 20000,:]
    # data_test = data.iloc[len(data) - 20000:len(data),:]
    add_label(data_train)
    trainset = data_train.iloc[:,:].values
    trainData_scaled = sc.fit_transform(trainset)
    
    label = data_train['Label']
    # y = ohe.fit(label)
    print('label : ')
    print(label)
    # open_values = trainset[:,-1]

    # temp = sc_.fit_transform(open_values.reshape(-1, 1))

    x_train = []
    y_train = []

    #! inja label haro ok mikonim :
    

    for i in range(length, len(trainData_scaled)):
        x_train.append(trainData_scaled[i-length:i,0:4])
        y_train.append(trainData_scaled[i, 4])
        

    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = ohe.fit_transform(y_train.reshape(-1, 1))
    x_train = np.reshape(x_train, (x_train.shape[0],1, 1, length, 4))
    # y_train = np.reshape(y_train, (y_train.shape[0], 2))
    return x_train, y_train


def add_label(df):
    idx = len(df.columns)
    new_col = np.where(df['close'] >= df['close'].shift(1), 1, 0)  
    df.insert(loc=idx, column='Label', value=new_col)
    df = df.fillna(0)


model = main_model()
print(model.summary())
dot_img_file = 'model.png'
plot_model(model, to_file = dot_img_file, show_shapes = True, show_layer_names = True, expand_nested = True)
data = read_data()
x_train, y_train = prepare_data(data, 60)

print(y_train)
print(y_train[-10:])
print(y_train.shape)