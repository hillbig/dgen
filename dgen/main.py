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
    parser.add_argument('--gpu', '-g', default=4, type=int, help='GPU Num')
    args = parser.parse_args()
    data = mnist.load_mnist_data()
    x = data['data'].astype(np.float32)
    x /= 255
    # y = data['target'].astype(np.int32)
    N = 60000
    x_train, x_test = np.split(x, [N])
    # y_train, y_test = np.split(y, [N])

    print "use {} gpus".format(args.gpu)

    models = [None] * args.gpu
    models[0] = vae.VAEBernoulli(x_dim=784, 
                                 z_dim=50,
                                 h_dim=200,
                                 n_layers=1,
                                 activate=F.tanh,
                                 use_dropout=False,
                                 use_bn=True,
                                 wmean=1.0,
                                 wlogvar=0.001)

    for i in xrange(0, args.gpu):
        if i > 0:
            models[i] = models[0].copy()

    for i in xrange(0, args.gpu):
        models[i].to_gpu(i)


    n_epoch = args.epoch
    batchsize = args.batchsize
    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-4)
    optimizer.setup(models[0])
    xp = cuda.cupy

    N_train = len(x_train)
    N_test = len(x_test)

    for step in xrange(7):
        n_epoch = 50 * 3 ** step
        optimizer.alpha = 0.001 * (10.0 ** (-step / 7.0))
        print "alpha=", optimizer.alpha
        for epoch in xrange(1, n_epoch+1):
            perm = np.random.permutation(N)
            train_loss_sum = 0
            start = time.time()
            for i in xrange(0, N_train, batchsize):
                with cuda.get_device(0):
                    x_batch = xp.asarray(x_train[perm[i:i+batchsize]])
                    r = xp.random.rand(*x_batch.shape)
                    xb = (x_batch > r).astype(xp.float32)
                losses =  [None] * args.gpu
                step = batchsize//args.gpu
                xb_gpu = [None] * args.gpu
                for j in xrange(args.gpu):
                    xb_gpu[j] = cuda.to_gpu(xb[j*step:(j+1)*step], j)

                for j in xrange(args.gpu):
                    with cuda.get_device(j):
                        losses[j] = models[j](xb_gpu[j])


                for j in xrange(args.gpu):
                    models[j].zerograds()
                    losses[j].backward()

                for j in xrange(1, args.gpu):
                    models[0].addgrads(models[j])
                
                optimizer.update()
                for j in xrange(1, args.gpu):
                    models[j].copyparams(models[0])

                for j in xrange(args.gpu):
                    train_loss_sum += losses[j].data.get()

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
