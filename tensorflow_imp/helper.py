# Author: zhangshulin 
# Email: zhangslwork@yeah.net 
# Date: 2018-04-18 08:00:36 
# Last Modified by: zhangshulin
# Last Modified Time: 2018-04-18 08:00:36 
# Author: zhangshulin 


DATA_PATH = './datasets/all_couplets.txt'
CUT_PATH = './datasets/all_cut_couplets.txt'


import jieba


def cut_dataset(input_path=DATA_PATH, output_path=CUT_PATH):
    input_file = open(input_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'a', encoding='utf-8')

    for input_line in input_file:
        cut_words = jieba.cut(input_line)
        cut_text = ' '.join(cut_words)
        output_file.write(cut_text)

    input_file.close()
    output_file.close()


def test_cut_dataset():
    cut_dataset(input_path=DATA_PATH, output_path=CUT_PATH)


if __name__ == '__main__':
    test_cut_dataset()
