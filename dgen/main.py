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

    models = [None] * 4
    models[0] = vae.VAEBernoulli(x_dim=784, 
                                 z_dim=100,
                                 h_dim=200,
                                 n_layers=1,
                                 activate=F.elu,
                                 use_dropout=False,
                                 use_bn=True,
                                 wmean=1.0,
                                 wlogvar=0.001)
    for i in xrange(0, 4):
        if i > 0:
            models[i] = models[0].copy()

    for i in xrange(0, 4):
        models[i].to_gpu(i)
        print models[i].encoder.in_0.W.data.device
        print models[i].decoder.in_0.W.data.device



    n_epoch = args.epoch
    batchsize = args.batchsize
    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-4)
    optimizer.setup(models[0])
    xp = np if args.gpu < 0 else cuda.cupy
    print xp

    N_train = len(x_train)
    N_test = len(x_test)

    for step in xrange(7):
        n_epoch = 3 ** step
        optimizer.alpha = 0.01 * (10.0 ** (-step / 7.0))
        print "alpha=", optimizer.alpha
        for epoch in xrange(1, n_epoch+1):
            perm = np.random.permutation(N)
            train_loss_sum = 0
            start = time.time()
            for i in xrange(0, N_train, batchsize):
                print i
                with cuda.get_device(0):
                    x_batch = xp.asarray(x_train[perm[i:i+batchsize]])
                    r = xp.random.rand(*x.shape)
                    xb = (x > r).astype(xp.float32)
                losses =  [None]*4
                step = batchsize//4
                xb_gpu = [None] * 4
                for j in xrange(4):
                    xb_gpu[j] = cuda.to_gpu(xb[j*step:(j+1)*step], j)

                for j in xrange(4):
                    with cuda.get_device(j):
                        losses[j] = models[j](xb_gpu[j])

                for j in xrange(4):
                    train_loss_sum += losses[j].data.get()

                for j in xrange(4):
                    models[j].zerograds()
                    losses[j].backward()

                for j in xrange(1, 4):
                    models[0].addgrads(models[j])
                
                optimizer.update()
                for j in xrange(1, 4):
                    models[j].copyparams(models[0])

            test_loss_sum = 0
            for i in xrange(0, N_test, batchsize):
                x = xp.asarray(x_train[perm[i:i+batchsize]])
                r = xp.random.rand(*x.shape)
                xb = (x > r).astype(xp.float32)
                loss = models[0](xb)
                test_loss_sum += loss.data
            
            train_loss_sum /= N_train
            test_loss_sum /= N_test
            print "{} {} {} {}".format(epoch, train_loss_sum, test_loss_sum, time.time() - start)
            

    

if __name__ == '__main__':
    main()
