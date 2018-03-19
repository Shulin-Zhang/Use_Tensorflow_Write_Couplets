# zhangshulin
# 2018-3-17
# e-mail: zhangslwork@yeah.net


import numpy as np 
from collections import Counter


class Couplets_data_generator:

    def __init__(self, file_path, vocabs_size=1000, max_len=20):
        self._vocabs_size = vocabs_size
        self._max_len = max_len

        with open(file_path, encoding='utf8') as f:
            self._total_couplets = f.readlines()
            self._total_couplets = [couplet for couplet in self._total_couplets
                                     if len(couplet.strip()) != 0]

        self._word2Index, self._index2word = self._create_words_dict()
        self._total_dataset = self._create_dataset()


    def get_words_dict(self):
        return self._word2Index, self._index2word


    def load_datasets(self, dev_test_size=4000, shuffle=True):
        assert dev_test_size * 2 < len(self._total_dataset)

        if shuffle:
            total_set = self._shuffle_dataset()
        else:
            total_set = self._total_dataset

        test_set = total_set[-dev_test_size:]
        dev_set = total_set[-(2 * dev_test_size): -dev_test_size]
        train_set = total_set[: -(2 * dev_test_size)]

        return train_set, dev_set, test_set


    def load_sample(self, size=100):
        assert size < len(self._total_couplets)

        total_set = self._shuffle_dataset()
        sample = total_set[:size]

        return sample    


    def _create_words_dict(self):
        word_counter = Counter()

        for line in self._total_couplets:
            for word in line:
                word_counter[word] += 1

        max_len = len(word_counter)

        words_arr = sorted(word_counter, reverse=True, key=word_counter.get)
        words_arr = words_arr[:min(self._vocabs_size, max_len)]
        words_arr[-1] = 'UNK'

        word2Index = {}
        index2word = {}

        for index, word in enumerate(words_arr):
            word2Index[word] = index 
            index2word[index] = word 

        return word2Index, index2word


    def _create_dataset(self):
        m = len(self._total_couplets)
        max_len = min(len(max(self._total_couplets, key=len)), self._max_len)
        total_dataset = np.zeros((m, max_len), dtype=np.int32)

        for line in range(m):
            for pos, word in enumerate(self._total_couplets[line]):
                index = self._word2Index.get(word, len(self._word2Index) - 1)

                if pos < self._max_len:
                    total_dataset[line, pos] =  index

        return total_dataset


    def _shuffle_dataset(self):
        return self._total_dataset[np.random.permutation(len(self._total_dataset))]


# TEST
if __name__ == '__main__':
    print('start')
    generator = Couplets_data_generator('./datasets/all_couplets.txt')
    print()
    print(generator._shuffle_dataset().shape)
    train, dev, test = generator.load_datasets(dev_test_size=20, shuffle=True)
    print(train.shape)
    print(dev.shape)
    print(test.shape)


