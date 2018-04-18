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


import jieba
from collections import Counter
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

    words_counter = Counter(text)
    words_sorted = sorted(words_counter, key=words_counter.get, reverse=True)

    index2word = list(enumerate(words_sorted))
    word2index = {word: index for index, word in index2word}
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


def test_cut_dataset():
    cut_dataset(input_path=DATA_PATH, output_path=CUT_PATH)


def test_divide_dataset():
    divide_dataset()


if __name__ == '__main__':
    # test_cut_dataset()
    test_divide_dataset()


