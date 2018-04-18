# Author: zhangshulin 
# Email: zhangslwork@yeah.net 
# Date: 2018-04-18 08:00:36 
# Last Modified by: zhangshulin
# Last Modified Time: 2018-04-18 08:00:36 


DATA_PATH = './datasets/all_couplets.txt'
CUT_PATH = './datasets/all_cut_couplets.txt'
TRAIN_PATH = './datasets/train_couplets.txt'
DEV_PATH = './datasets/dev_couplets.txt'
TEST_PATH = './datasets/test_couplets.txt'
TRAIN_SET = './datasets/train_set.npy'
DEV_SET = './datasets/dev_set.npy'
TEST_SET = './datasets/test_set.npy'


import jieba
from collections import Counter
import numpy as np
import os


def cut_dataset(input_path=DATA_PATH, output_path=CUT_PATH):
    input_file = open(input_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'a', encoding='utf-8')

    for input_line in input_file:
        cut_words = jieba.cut(input_line)
        cut_text = ' '.join(cut_words)
        output_file.write(cut_text)

    input_file.close()
    output_file.close()


def create_vocab(input_path=CUT_PATH):
    with open(input_path, encoding='utf-8') as f:
        text = f.read()

    words_counter = Counter(text.split())
    words_sorted = [' '] + sorted(words_counter, key=words_counter.get, reverse=True)

    index2word = dict(enumerate(words_sorted))
    word2index = {word: index for index, word in index2word.items()}
    vocab_size = len(words_sorted)

    return vocab_size, index2word, word2index


def divide_dataset(all_path=CUT_PATH, train_path=TRAIN_PATH, dev_path=DEV_PATH, test_path=TEST_PATH, 
                   dev_size=4000, test_size=4000):
    all_file = open(all_path, 'r', encoding='utf-8')
    train_file = open(train_path, 'a', encoding='utf-8')
    dev_file = open(dev_path, 'a', encoding='utf-8') 
    test_file = open(test_path, 'a', encoding='utf-8')

    for index, line in enumerate(all_file):
        if index < dev_size:
            dev_file.write(line)
        elif index < dev_size + test_size:
            test_file.write(line)
        else:
            train_file.write(line)

    all_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()


def convert_text_line_to_ints(text_line, word2index):
    words_list = text_line.split()
    int_list = [word2index[word] for word in words_list]

    return int_list


def convert_data_file(file_path, word2index):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_lines_list = f.readlines()
    
    int_lines_list = [convert_text_line_to_ints(line, word2index) for line in text_lines_list]
    
    max_length = 0
    for int_line in int_lines_list:
        if len(int_line) > max_length:
            max_length = len(int_line)

    padding_array = np.zeros((len(int_lines_list), max_length), dtype=np.int32)
    for index, int_line in enumerate(int_lines_list):
        padding_array[index, :len(int_line)] = int_line
    
    return padding_array


def process_dataset(all_couplets=DATA_PATH):
    if not os.path.exists(CUT_PATH):
        cut_dataset()

    vocab_size, index2word, word2index = create_vocab()

    if os.path.exists(TRAIN_SET) and os.path.exists(DEV_SET) and os.path.exists(TEST_SET):
        train_set = np.load(TRAIN_SET)
        dev_set = np.load(DEV_SET)
        test_set = np.load(TEST_SET)

        print('vocab_size:', vocab_size)
        print('train_set shape:', train_set.shape)
        print('dev_set shape:', dev_set.shape)
        print('test_set shape:', test_set.shape)

        return vocab_size, index2word, word2index, train_set, dev_set, test_set

    if (not os.path.exists(TRAIN_PATH)) or (not os.path.exists(DEV_PATH)) or (not os.path.exists(TEST_PATH)):
        divide_dataset()
    
    train_set = convert_data_file(TRAIN_PATH, word2index)
    dev_set = convert_data_file(DEV_PATH, word2index)
    test_set = convert_data_file(TEST_PATH, word2index)

    np.save(TRAIN_SET, train_set)
    np.save(DEV_SET, dev_set)
    np.save(TEST_SET, test_set)

    print('vocab_size:', vocab_size)
    print('train_set shape:', train_set.shape)
    print('dev_set shape:', dev_set.shape)
    print('test_set shape:', test_set.shape)

    return vocab_size, index2word, word2index, train_set, dev_set, test_set


def test_cut_dataset():
    cut_dataset(input_path=DATA_PATH, output_path=CUT_PATH)


def test_divide_dataset():
    divide_dataset()


def test_process_dataset():
    process_dataset()


if __name__ == '__main__':
    # test_cut_dataset()
    # test_divide_dataset()
    test_process_dataset()


