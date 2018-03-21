# zhangshulin
# 2018-3-19
# e-mail: zhangslwork@yeah.net

FILF_PATH = './datasets/all_couplets.txt'


from datasets_creator import Datasets_creator
from batch_generator import Batch_generator
import numpy as np


def load_datasets(vocabs_size=5000, max_len=30, dev_test_size=4000, batch_size=32, n_a=16):
    generator = Datasets_creator(FILF_PATH, vocabs_size, max_len)

    train_set, dev_set, test_set = generator.load_datasets(dev_test_size=dev_test_size)
    train_generator = Batch_generator(train_set, batch_size, vocabs_size, n_a)
    dev_generator = Batch_generator(dev_set, batch_size, vocabs_size, n_a)
    test_generator = Batch_generator(test_set, batch_size, vocabs_size, n_a)

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


def load_sample_datasets(vocabs_size=5000, max_len=30, batch_size=16, sample_size=100, n_a=16):
    generator = Datasets_creator(FILF_PATH, vocabs_size, max_len)

    sample_set = generator.load_sample(size=sample_size)
    sample_generator = Batch_generator(sample_set, batch_size, vocabs_size, n_a)

    word2index, index2word = generator.get_words_dict()

    return {
        'sample_gen': sample_generator,
        'word2index': word2index,
        'index2word': index2word,
        'vocabs_size': vocabs_size,
        'max_len': max_len 
    }


def convert_predict_to_text(predict, index2word):
    sequence = np.argmax(predict, axis=-1)
    predict[sequence == 1] = 0
    result_seq = np.argmax(predict, axis=-1)

    return convert_sequence_to_text(result_seq, index2word)


def convert_sequence_to_text(sequence, index2word):
    text_arr = [index2word[int(index)] for index in np.squeeze(sequence)]

    return ''.join(text_arr)
    

def convert_text_to_onehot(text, vocabs_size, max_len, word2index):
    onehot = np.zeros((1, max_len, vocabs_size))
    for pos, word in enumerate(text.strip()):
        index = word2index[word]
        onehot[0, pos + 1, index] = 1

    return onehot


# TEST
if __name__ == '__main__':
    result = load_sample_datasets()
    generator = result['sample_gen']
    word2index = result['word2index']

    num = 0
    for x, y in generator:
        print(len(x), y[0].shape)
        num += 1
        if num == 20:
            break

    text = '床前明月光'
    onehot = convert_text_to_onehot(text, 5000, 30, word2index)
    print('onehot:', onehot.shape)
            

