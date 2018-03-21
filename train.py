# zhangshulin
# 2018-3-21
# e-mail: zhangslwork@yeah.net


VOCABS_SIZE = 5000
LSTM_NA = 128
MAX_LEN = 30
DEV_TEST_SIZE = 4000
BATCH_SIZE = 64
EPOCHS = 10
WEIGHTS = './weights.h5'


from couplets_utils import load_datasets
from model import create_train_model
import numpy as np
import os


print('loading datasets')
datasets = load_datasets(VOCABS_SIZE, MAX_LEN, DEV_TEST_SIZE, BATCH_SIZE, LSTM_NA)

train_gen = datasets['train_gen']
dev_gen = datasets['dev_gen']
test_gen = datasets['test_gen']
word2index = datasets['word2index']
index2word = datasets['index2word']

print('begin creating model.')
train_model = create_train_model(VOCABS_SIZE, LSTM_NA, MAX_LEN)

train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('model creating complete')

print('begin load weights')

if os.path.exists(WEIGHTS):
    train_model.load_weights(WEIGHTS)
print('weight load complete')

print('begin training')
train_model.fit_generator(train_gen, epochs=EPOCHS)
train_model.save_weights(WEIGHTS)
print('training complete')