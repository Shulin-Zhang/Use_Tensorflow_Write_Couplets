# zhangshulin
# 2018-3-19
# e-mail: zhangslwork@yeah.net

FILF_PATH = './datasets/all_couplets.txt'


from datasets_creator import Datasets_creator
from batch_generator import Batch_generator
import numpy as np


def load_datasets(vocabs_size=5000, max_len=30, dev_test_size=4000, batch_size=32):
    generator = Datasets_creator(FILF_PATH, vocabs_size, max_len)

    train_set, dev_set, test_set = generator.load_datasets(dev_test_size=dev_test_size)
    train_generator = Batch_generator(train_set, batch_size, vocabs_size)
    dev_generator = Batch_generator(dev_set, batch_size, vocabs_size)
    test_generator = Batch_generator(test_set, batch_size, vocabs_size)

    word2index, index2word = generator.get_words_dict()

    return {
        'train_gen': train_generator,
        'dev_gen': dev_generator,
        'test_gen': test_generator,
        'word2index': word2index,
        'index2word': index2word,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def load_sample_datasets(vocabs_size=5000, max_len=30, batch_size=16, sample_size=100):
    generator = Datasets_creator(FILF_PATH, vocabs_size, max_len)

    sample_set = generator.load_sample(size=sample_size)
    sample_generator = Batch_generator(sample_set, batch_size, vocabs_size)

    word2index, index2word = generator.get_words_dict()

    return {
        'sample_gen': sample_generator,
        'word2index': word2index,
        'index2word': index2word,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def convert_sequence_to_text(sequence, index2word):
    text_arr = [index2word[index] for index in sequence]

    return ''.join(text_arr)



# TEST
if __name__ == '__main__':
    result = load_sample_datasets()
    generator = result['sample_gen']

    num = 0
    for x, y in generator:
        print(x.shape, y.shape)
        num += 1
        if num == 20:
            break
            

