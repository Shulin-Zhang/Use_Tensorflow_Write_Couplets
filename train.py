# zhangshulin
# 2018-3-21
# e-mail: zhangslwork@yeah.net


VOCABS_SIZE = 5000
LSTM_NA = 128
MAX_LEN = 30
DEV_TEST_SIZE = 4000
WEIGHTS = './weights.h5'


from couplets_utils import load_datasets, load_sample_datasets
from model import create_train_model
from keras.optimizers import Adam 
import numpy as np
import os
import logging
import argparse


def train(epochs=1, learning_rate=0.01, batch_size=64, resume=True, sample=False, mode='train'):
    logging.info('loading datasets')

    if sample:
        dataset = load_sample_datasets(VOCABS_SIZE, MAX_LEN, batch_size, 1000, LSTM_NA)
        generator = dataset['sample_gen']
    else:
        datasets = load_datasets(VOCABS_SIZE, MAX_LEN, DEV_TEST_SIZE, batch_size, LSTM_NA)
        if mode == 'train':
            generator = datasets['train_gen']
        elif mode == 'evaluate':
            generator = datasets['dev_gen']
        else:
            generator = datasets['test_gen']

    logging.info('begin creating model.')
    model = create_train_model(VOCABS_SIZE, LSTM_NA, MAX_LEN)

    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info('model creating complete')

    logging.info('begin load weights')

    if os.path.exists(WEIGHTS) and resume:
        model.load_weights(WEIGHTS)
        logging.info('weight load complete')

    if mode == 'train':
        logging.info('begin training')
        model.fit_generator(generator, epochs=epochs)
        model.save_weights(WEIGHTS)
        logging.info('training end')
    elif mode == 'evaluate':
        logging.info('begin evaluate')
        evaluation = model.evaluate_generator(generator)
        accuracy = sum(evaluation[-VOCABS_SIZE:]) / VOCABS_SIZE 
        print('total loss: {}, average accuray: {}'.format(evaluation[0], accuracy))
        logging.info('evaluating end')
    else:
        logging.info('begin test')
        evaluation = model.evaluate_generator(generator)
        accuracy = sum(evaluation[-VOCABS_SIZE:]) / VOCABS_SIZE 
        print('total loss: {}, average accuray: {}'.format(evaluation[0], accuracy))
        logging.info('testing end')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch', default=64, type=int, help='mini batch size')
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--sample', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    
    args = parser.parse_args()
    
    if args.evaluate == True:
        mode = 'evaluate'
    elif args.test == True:
        mode = 'test'
    else:
        mode = 'train'

    train(
        epochs=args.epochs, 
        learning_rate=args.lr, 
        batch_size=args.batch, 
        resume=args.resume, 
        sample=args.sample,
        mode=mode
    )

