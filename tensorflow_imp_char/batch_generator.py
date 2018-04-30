# Author: zhangshulin 
# Email: zhangslwork@yeah.net 
# Date: 2018-04-30  
# Last Modified by: zhangshulin


import tensorflow as tf
import numpy as np


class Batch_generator:

    def __init__(self, set_array, shuffle=True, buffer_size=10000):
        self._data = tf.data.Dataset.from_tensor_slices(set_array)
        self.shape = set_array.shape
        
        if shuffle:
            self._data = self._data.shuffle(buffer_size=buffer_size)


    def get_batch(self, session, batch_size, epochs):
        batch_data = self._data.batch(batch_size)
        repeat_data = batch_data.repeat(epochs)
        
        iterator = repeat_data.make_one_shot_iterator()
        next = iterator.get_next()

        while True:
            try:
                data_set = session.run(next)
                Y = data_set
                fore_zeros = np.zeros((data_set.shape[0], 1), dtype=np.int32)
                X = np.concatenate((fore_zeros, Y[:, : -1]), axis=1)

                yield X, Y

            except tf.errors.OutOfRangeError:
                break


def test():
    from datasets_creator import Datasets_creator 

    generator = Datasets_creator('./datasets/all_couplets.txt')
    _, _, test_set = generator.load_datasets(dev_test_size=4000, shuffle=True)

    data_g = Batch_generator(test_set)
    sess = tf.Session()
    batch_g = data_g.get_batch(sess, 128, 1)

    step = 0
    for X, Y in batch_g:
        print('step: ', step)
        print('X: ', X.shape, X[0, :5])
        print('Y: ', Y.shape, Y[0, :5])
        step += 1


if __name__ == '__main__':
    test()
