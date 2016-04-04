import mnist
import numpy as np
import vae
import argparse
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--batchsize', '-b', default=20, type=int, help='mini batch size')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value is for CPU')
    args = parser.parse_args()
    data = mnist.load_mnist_data()
    x = data['data'].astype(np.float32)
    x /= 255
    # y = data['target'].astype(np.int32)
    N = 60000
    x_train, x_test = np.split(x, [N])
    # y_train, y_test = np.split(y, [N])

    model = vae.VAEBernoulli(x_dim=784, 
                             z_dim=100,
                             h_dim=200,
                             n_layers=1,
                             activate=F.relu,
                             use_dropout=False,
                             use_bn=True,
                             wmean=1.0,
                             wlogvar=0.001)

    print "gpu=", args.gpu
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    n_epoch = args.epoch
    batchsize = args.batchsize
    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-4)
    optimizer.setup(model)
    xp = np if args.gpu < 0 else cuda.cupy
    print xp

    N_train = len(x_train)
    N_test = len(x_test)
    for epoch in xrange(1, n_epoch+1):
        start = time.time()
        perm = np.random.permutation(N)
        train_loss_sum = 0
        for i in xrange(0, N_train, batchsize):
            x = xp.asarray(x_train[perm[i:i+batchsize]])
            r = xp.random.rand(*x.shape)
            xb = (x > r).astype(xp.float32)
            loss = model(xb)
            train_loss_sum += loss.data
            loss /= batchsize
            optimizer.zero_grads()
            loss.backward()
            optimizer.update()
        test_loss_sum = 0
        for i in xrange(0, N_test, batchsize):
            x = xp.asarray(x_train[perm[i:i+batchsize]])
            r = xp.random.rand(*x.shape)
            xb = (x > r).astype(xp.float32)
            loss = model(xb)
            test_loss_sum += loss.data
            
        train_loss_sum /= N_train
        test_loss_sum /= N_test
        print "{} {} {} {}".format(epoch, train_loss_sum, test_loss_sum, time.time() - start)
            

    

if __name__ == '__main__':
    main()
