# Author: zhangshulin 
# Email: zhangslwork@yeah.net 
# Date: 2018-04-30  
# Last Modified by: zhangshulin

FILF_PATH = './datasets/all_couplets.txt'


from datasets_creator import Datasets_creator
from batch_generator import Batch_generator
import numpy as np


def load_datasets(max_vocabs_size=5000, max_len=30, dev_test_size=4000):
    creator = Datasets_creator(FILF_PATH, max_vocabs_size, max_len)

    train_set, dev_set, test_set = creator.load_datasets(dev_test_size=dev_test_size)
    train_generator = Batch_generator(train_set)
    dev_generator = Batch_generator(dev_set)
    test_generator = Batch_generator(test_set)

    char2index, index2char = creator.get_chars_dict()

    return {
        'train_gen': train_generator,
        'dev_gen': dev_generator,
        'test_gen': test_generator,
        'char2index': char2index,
        'index2char': index2char,
        'vocabs_size': len(char2index),
        'max_len': max_len 
    }


def load_sample_datasets(vocabs_size=5000, max_len=30, sample_size=100):
    creator = Datasets_creator(FILF_PATH, vocabs_size, max_len)

    sample_set = creator.load_sample(size=sample_size)
    sample_generator = Batch_generator(sample_set)

    char2index, index2char = creator.get_chars_dict()

    return {
        'sample_gen': sample_generator,
        'char2index': char2index,
        'index2char': index2char,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def load_chars_dict(vocabs_size=5000, max_len=30, n_a=16):
    creator = Datasets_creator(FILF_PATH, vocabs_size, max_len)
    char2index, index2char = creator.get_chars_dict()

    return char2index, index2char


def convert_sequence_to_text(sequence, index2char):
    text_arr = [index2char[int(index)] for index in np.squeeze(sequence)]

    return ''.join(text_arr)
    

# TEST
if __name__ == '__main__':
    import tensorflow as tf

    result = load_datasets()
    generator = result['dev_gen']
    char2index = result['char2index']

    sess = tf.Session()

    step = 0
    for x, y in generator.get_batch(sess, 5, 1):
        print('x:', x.shape, 'y:', y.shape)
        if step == 5:
            break
        step += 1
            

