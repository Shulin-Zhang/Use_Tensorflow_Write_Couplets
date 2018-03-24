# zhangshulin
# 2018-3-21
# e-mail: zhangslwork@yeah.net


from keras.layers import LSTM, Dense, Lambda, Reshape, Input, Dropout
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def create_train_model(n_x, n_a, Tx, keep_prob=1):    
    input = Input(shape=(Tx, n_x), name='x0')
    a_0_0 = Input(shape=(n_a,), name='a_0_0')
    c_0_0 = Input(shape=(n_a,), name='c_0_0')
    a_1_0 = Input(shape=(n_a,), name='a_1_0')
    c_1_0 = Input(shape=(n_a,), name='c_1_0')
        
    a_0 = a_0_0
    c_0 = c_0_0
    a_1 = a_1_0
    c_1 = c_1_0
    
    lstm_cell_0 = LSTM(units=n_a, return_state=True, name='lstm_0')
    lstm_cell_1 = LSTM(units=n_a, return_state=True, name='lstm_1')
    dense_layer_2 = Dense(units=n_x, activation='softmax', name='softmax_2')
        
    outputs = []
        
    for i in range(Tx):
        x = Lambda(lambda j: j[:, i, :])(input)
        x = Reshape(target_shape=(1, -1))(x)
        a_0, x, c_0 = lstm_cell_0(x, initial_state=[a_0, c_0])
        x = Dropout(rate=1-keep_prob)(x)
        x = Reshape(target_shape=(1, -1))(x)
        a_1, x, c_1 = lstm_cell_1(x, initial_state=[a_1, c_1])
        x = Dropout(rate=1-keep_prob)(x)
        x = dense_layer_2(x)
        outputs.append(x)
                            
    model = Model(inputs=[input, a_0_0, c_0_0, a_1_0, c_1_0], outputs=outputs)
        
    return model


def create_infer_model(n_x, n_a, Tx):
    input = Input(shape=(Tx, n_x), name='x0')
    a_0_0 = Input(shape=(n_a,), name='a_0_0')
    c_0_0 = Input(shape=(n_a,), name='c_0_0')
    a_1_0 = Input(shape=(n_a,), name='a_1_0')
    c_1_0 = Input(shape=(n_a,), name='c_1_0')
    
    x = Lambda(lambda x: x[:, 0, :])(input)
    a_0 = a_0_0
    c_0 = c_0_0
    a_1 = a_1_0
    c_1 = c_1_0

    lstm_cell_0 = LSTM(units=n_a, return_state=True, name='lstm_0')
    lstm_cell_1 = LSTM(units=n_a, return_state=True, name='lstm_1')
    dense_layer_2 = Dense(units=n_x, activation='softmax', name='softmax_2')
    
    def one_hot(x):
        x = K.argmax(x)
        x = tf.one_hot(x, n_x) 
        return x
    
    def select_x(x, i):
        return tf.cond(
            tf.equal(tf.reduce_sum(input[:, i+1, :]), 0),
            lambda : x,
            lambda : input[:, i+1, :]
        )
        
    def select_output(x, i):
        return tf.cond(
            tf.less(i + 1, Tx),
            lambda : select_x(x, i),
            lambda : x
        )
    
    outputs = []
    
    for i in range(Tx - 1):              
        x = Reshape(target_shape=(1, -1))(x)
        a_0, x, c_0 = lstm_cell_0(x, initial_state=[a_0, c_0])
        x = Reshape(target_shape=(1, -1))(x)
        a_1, x, c_1 = lstm_cell_1(x, initial_state=[a_1, c_1])
        x = dense_layer_2(x)
        x = Lambda(one_hot)(x)
        
        x = Lambda(lambda x: select_output(x, i))(x)
        outputs.append(x)
        
    model = Model(inputs=[input, a_0_0, c_0_0, a_1_0, c_1_0], outputs=outputs)
    
    return model

