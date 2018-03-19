#encoding:utf-8
# zhangshulin
# 2018-3-19
# e-mail: zhangslwork@yeah.net

FILF_PATH = './datasets/all_couplets.txt'


from data_generator import Couplets_data_generator
import numpy as np


def load_datasets(vocabs_size=5000, max_len=30, dev_test_size=4000):
    generator = Couplets_data_generator(FILF_PATH, vocabs_size, max_len)

    train_set, dev_set, test_set = generator.load_datasets(dev_test_size=dev_test_size)

    train_X, train_Y = _create_X_Y(train_set)
    dev_X, dev_Y = _create_X_Y(dev_set)
    test_X, test_Y = _create_X_Y(test_set)

    word2index, index2word = generator.get_words_dict()

    return {
        'train_X': train_X,
        'train_Y': train_Y,
        'dev_X': dev_X,
        'dev_Y': dev_Y,
        'test_X': test_X,
        'test_Y': test_Y,
        'word2index': word2index,
        'index2word': index2word,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def load_sample_datasets(vocabs_size=5000, max_len=30):
    generator = Couplets_data_generator(FILF_PATH, vocabs_size, max_len)
    sample_set = generator.load_sample(size=100)

    sample_X, sample_Y = _create_X_Y(sample_set)
    word2index, index2word = generator.get_words_dict()

    return {
        'sample_X': sample_X,
        'sample_Y': sample_Y,
        'word2index': word2index,
        'index2word': index2word,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def convert_sequence_to_text(sequence, index2word):
    text_arr = [index2word[index] for index in sequence]

    return ''.join(text_arr)



def _create_X_Y(data_set):
    m = data_set.shape[0]
    zeros_start = np.zeros((m, 1))
    
    X = np.concatenate([zeros_start, data_set], axis=1)
    X = X[:, :-1]
    Y = data_set

    return X, Y


# TEST
if __name__ == '__main__':
    result = load_sample_datasets()
    sample_X = result['sample_X']
    sample_Y = result['sample_Y']
    index2word = result['index2word']
    print('X: ', sample_X.shape)
    print('Y: ', sample_Y.shape)
    print('X[0]',sample_X[0])
    print('Y[0]', sample_Y[0])
    print('text: ', convert_sequence_to_text(sample_Y[0], index2word))

