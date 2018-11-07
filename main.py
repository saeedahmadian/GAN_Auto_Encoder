import tensorflow as tf
from config import args
import numpy as np

from model import CyberGAN
from data import CyberData


def main(_):

    data = CyberData(args)
    # 'Here you input your Data in the numpy format pay attantion your data must fit the config part '
    data.X = np.load('real_data.npy')
    data.X_Auto = np.load('desired_data.npy')

    model = CyberGAN(args)

    if args.mode == 'train':
        model.train(data)
    if args.mode == 'test':
        model.test(data)


if __name__ == '__main__':
    tf.app.run()