import os


def main():
    # one can find this dir in the config out file
    log_dir = './checkpoints/5016_16568_241014-124156/tfb_train'
    os.system('tensorboard --logdir={}'.format(log_dir))
    return


if __name__ == '__main__':
    main()
