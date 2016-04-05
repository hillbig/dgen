import chainer.functions as F
import chainer.links as L
import chainer
import time
import WNLinear

def _in(i):
    return 'in_{}'.format(i)

def _fc(i):
    return 'fc_{}'.format(i)

def _out(i):
    return 'out_{}'.format(i)

def _bn(i):
    return 'bn_{}'.format(i)

class MLP(chainer.Chain):
    def __init__(self, in_shapes, out_shapes, n_units, out_wscales, n_layers, activate, use_dropout, use_bn):
        assert isinstance(in_shapes, tuple)
        assert isinstance(out_shapes, tuple)
        assert isinstance(out_wscales, tuple)
        assert len(out_shapes) == len(out_wscales)

        self.in_shapes = in_shapes
        self.out_shapes = out_shapes
        self.n_units = n_units
        self.n_layers = n_layers
        self.activate = activate
        self.use_dropout = use_dropout
        self.use_bn = use_bn

        layers = {}

        for i, shape in enumerate(in_shapes):
            layers[_in(i)] = WNLinear.WNLinear(shape, n_units)

        for i in xrange(n_layers):
            layers[_fc(i)] = WNLinear.WNLinear(n_units, n_units)

        for i, shape in enumerate(out_shapes):
            layers[_out(i)] = WNLinear.WNLinear(n_units, shape, wscale=out_wscales[i])

        if use_bn:
            for i in xrange(n_layers+1):
                layers[_bn(i)] = L.BatchNormalization(n_units)

        super(MLP, self).__init__(**layers)
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    def __call__(self, xs, test=False):
        assert len(xs) == len(self.in_shapes)
        h = sum(self[_in(i)](x) for i, x in enumerate(xs))

        if self.use_bn:
            h = self[_bn(0)](h)
        h = self.activate(h)

        for i in xrange(self.n_layers):
            h = self[_fc(i)](h)
            if self.use_bn:
                h = self[_bn(i+1)](h)
            h = self.activate(h)
            if self.use_dropout:
                h = F.dropout(h, train=self.train)

        outputs = tuple(self[_out(i)](h) for i in xrange(len(self.out_shapes)))

        return outputs
        

                 
