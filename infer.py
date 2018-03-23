# zhangshulin
# 2018-3-21
# e-mail: zhangslwork@yeah.net


VOCABS_SIZE = 5000
LSTM_NA = 128
MAX_LEN = 30
BATCH_SIZE = 64
WEIGHTS = './weights.h5'


from model import create_infer_model
from couplets_utils import convert_text_to_onehot, convert_predict_to_text, load_words_dict
import sys
import numpy as np
import argparse


def write_couplets(begin_text, infer_model, word2index, index2word):
    x = convert_text_to_onehot(begin_text, VOCABS_SIZE, MAX_LEN, word2index)
    a0 = np.zeros((1, LSTM_NA))
    c0 = np.zeros((1, LSTM_NA))
    
    result = infer_model.predict([x, a0, c0])
    result_text = convert_predict_to_text(np.array(result), index2word)
    
    return result_text


def infer(begin_text):
    word2index, index2word = load_words_dict(VOCABS_SIZE, MAX_LEN, LSTM_NA)

    infer_model = create_infer_model(VOCABS_SIZE, LSTM_NA, MAX_LEN)
    infer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    infer_model.load_weights(WEIGHTS)

    result_text = write_couplets(begin_text, infer_model, word2index, index2word)

    print(result_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, default='', help='请输入对联的开始文本')
    args = parser.parse_args()
    infer(args.text)

