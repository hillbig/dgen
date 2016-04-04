import chainer
from chainer import Variable
import chainer.functions as F
import mlp

class VAEGaussian(chainer.Chain):
    def __init__(self, x_dim, z_dim, h_dim, n_layers,
                 activate, use_dropout, use_bn, wmean, wlogvar):
        self.loss_fun = loss_fun
        super(VAEGaussian, self).__init__(
            encoder = mlp.MLP(
                in_shapes=(x_dim,),
                out_shapes=(z_dim, z_dim,),
                n_units=h_dim,
                out_wscales=(wmean, wlogvar),
                n_layers=n_layers,
                activate=activate,
                use_dropout=use_dropout,
                use_bn=use_bn),
            decoder = mlp.MLP(
                in_shapes=(z_dim,),
                out_shapes=(x_dim, x_dim,),
                n_units=h_dim,
                out_wscales=(wmean, wlogvar),
                n_layers=n_layers,
                activate=activate,
                use_dropout=use_dropout,
                use_bn=use_bn))

    def __call__(self, x):
        xp = self.encoder.xp
        x = Variable(xp.asarray(x))
        zm, zv = self.encoder((x,))
        z = F.gaussian(zm, zv)
        mean, ln_var = self.decoder((z,))
        kl_loss = F.gaussian_kl_divergence(zm, zv)
        nll_loss = F.gaussian_nll(x, mean, ln_var)
        loss = kl_loss + nll_loss
        return loss

class VAEBernoulli(chainer.Chain):
    def __init__(self, x_dim, z_dim, h_dim, n_layers,
                 activate, use_dropout, use_bn, wmean, wlogvar):
        super(VAEBernoulli, self).__init__(
            encoder = mlp.MLP(
                in_shapes=(x_dim,),
                out_shapes=(z_dim, z_dim,),
                n_units=h_dim,
                out_wscales=(wmean, wlogvar),
                n_layers=n_layers,
                activate=activate,
                use_dropout=use_dropout,
                use_bn=use_bn),
            decoder = mlp.MLP(
                in_shapes=(z_dim,),
                out_shapes=(x_dim,),
                n_units=h_dim,
                out_wscales=(wmean,),
                n_layers=n_layers,
                activate=activate,
                use_dropout=use_dropout,
                use_bn=use_bn))

    def __call__(self, x):
        xp = self.encoder.xp
        x = Variable(xp.asarray(x))
        zm, zv = self.encoder((x,))
        z = F.gaussian(zm, zv)
        y = self.decoder((z,))[0]
        kl_loss = F.gaussian_kl_divergence(zm, zv)
        nll_loss = F.bernoulli_nll(x, y)
        loss = kl_loss + nll_loss
        return loss
