# zhangshulin
# 2018-3-17
# e-mail: zhangslwork@yeah.net


TRAIN_IN_PATH = './datasets/rawdata/train/in.txt'
TRAIN_OUT_PATH = './datasets/rawdata/train/out.txt'
TEST_IN_PATH = './datasets/rawdata/test/in.txt'
TEST_OUT_PATH = './datasets/rawdata/test/out.txt'
TOTAL_PATH = './datasets/all_couplets.txt'


def create_data_file(train_in_path, train_out_path, test_in_path, test_out_path, total_path):
    with open(train_in_path, 'r', encoding='utf8') as f:
        train_in_arr = f.readlines()

    with open(train_out_path, 'r', encoding='utf8') as f:
        train_out_arr = f.readlines()

    with open(test_in_path, 'r', encoding='utf8') as f:
        test_in_arr = f.readlines()

    with open(test_out_path, 'r', encoding='utf8') as f:
        test_out_arr = f.readlines()

    train_in_arr = map(process_in_couplet, train_in_arr)
    train_out_arr = map(process_out_couplet, train_out_arr)
    test_in_arr = map(process_in_couplet, test_in_arr)
    test_out_arr = map(process_out_couplet, test_out_arr)

    train_in_out_arr = [up + down for up, down in zip(train_in_arr, train_out_arr) 
                        if len(up.strip()) != 0 and len(down.strip()) != 0]
    test_in_out_arr = [up + down for up, down in zip(test_in_arr, test_out_arr)
                        if len(up.strip()) != 0 and len(down.strip()) != 0]

    total_arr = train_in_out_arr + test_in_out_arr

    with open(total_path, 'w', encoding='utf8') as f:
        f.writelines(total_arr)

    print('data file creating complete ^_^')


def process_in_couplet(couplet):
    return couplet.replace(' ', '').replace('\n', '；')


def process_out_couplet(couplet):
    return couplet.replace(' ', '').replace('\n', '。\n')


if __name__ == '__main__':
    create_data_file(TRAIN_IN_PATH, TRAIN_OUT_PATH, TEST_IN_PATH, TEST_OUT_PATH, TOTAL_PATH)