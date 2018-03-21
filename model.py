# zhangshulin
# 2018-3-21
# e-mail: zhangslwork@yeah.net


from keras.layers import LSTM, Dense, Lambda, Reshape, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def create_train_model(n_x, n_a, Tx):
    input = Input(shape=(Tx, n_x), name='x0')
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
        
    a = a0
    c = c0
    
    lstm_cell = LSTM(units=n_a, return_state=True, name='lstm_0')
    dense_layer = Dense(units=n_x, activation='softmax', name='softmax_1')
        
    outputs = []
        
    for i in range(Tx):
        x = Lambda(lambda j: j[:, i, :])(input)
        x = Reshape(target_shape=(1, -1))(x)
        a, x, c = lstm_cell(x, initial_state=[a, c])
        x = dense_layer(x)
        outputs.append(x)
                            
    model = Model(inputs=[input, a0, c0], outputs=outputs)
        
    return model



def create_infer_model(n_x, n_a, Tx):
    tf.reset_default_graph()
    
    X = Input(shape=(Tx, n_x), name='input_X')
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    
    x = Lambda(lambda x: x[:, 0, :])(X)
    a = a0
    c = c0

    lstm_cell = LSTM(units=n_a, return_state=True, name='lstm_0')
    dense_layer = Dense(units=n_x, activation='softmax', name='softmax_1')
    
    def one_hot(x):
        x = K.argmax(x)
        x = tf.one_hot(x, n_x) 
        return x
    
    def select_x(x, i):
        return tf.cond(
            tf.equal(tf.reduce_sum(X[:, i+1, :]), 0),
            lambda : x,
            lambda : X[:, i+1, :]
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
        a, x, c = lstm_cell(x, initial_state=[a, c])
        x = dense_layer(x)
        x = Lambda(one_hot)(x)
        
        x = Lambda(lambda x: select_output(x, i))(x)
        outputs.append(x)
        
        
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model

