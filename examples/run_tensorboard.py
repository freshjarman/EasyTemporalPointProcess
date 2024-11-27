import os


def main():
    # one can find this dir in the config out file
    log_dir = './checkpoints/9892_12116_241014-130341/tfb_train'
    os.system('tensorboard --logdir={}'.format(log_dir))
    return


if __name__ == '__main__':
    main()
