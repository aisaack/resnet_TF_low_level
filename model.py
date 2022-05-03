import tensorflow as tf
from custmm_layers import (Conv2d, FC, BatchNorm)

class ResBlockLow(tf.keras.Model):
    def __init__(
        self,
        stage,
        block,
        filter,
        stride,
        name = 'ResBlock',
        **kwarg
    ):
        super(ResBlockLow, self).__init__(name = name + f'{stage}-{block}', **kwargs)
        self.stride = stride
        self.bn1 = BatchNorm()
        self.conv1 = Conv2d(filters = filter,
                            kernel_size = (3, 3),
                            strides = stride,
                            padding = 'same')
        self.bn2 = BatchNorm()
        self.conv2 = Conv2d(filters = filter,
                            kernel_size = (3, 3),
                            strides = (1, 1),
                            padding = 'same')

        if stride != 1:
            self.id = Conv2d(filters = filter,
                             kernel_size = (1, 1),
                             strides = (2, 2),
                             padding = 'valid')

    def call(self, X):
        x = tf.nn.relu(self.conv1(self.bn1(X)))
        x = tf.nn.relu(self.conv2(self.bn2(x)))
        if self.stride != 1:
            id = self.id(X)
            return x + id
        else:
            return x + X


class ResNetLow(tf.keras.Model):
    def __init__(
        self,
        architecture,
        name = 'ResNet',
        **kwargs
    ):
        super(ResNetLow, self).__init__(name = name, **kwargs)
        self.architecture = architecture
        self.net_layers = {}
        self.net_layers['network_stem'] = Conv2d(filters = architecture.get('filters')[0],
                                                 kernel_size = (3, 3),
                                                 strides = (1, 1),
                                                 padding = 'same')
        for stage, (filter, stride, blocks) in enumerate(zip(architecture.get('filters'),
                                                            architecture.get('strides'),
                                                            architecture.get('blocks'))):
            for block in range(blocks):
                if block != 0:
                    stride = 1
                self.net_layers[f'{stage}-{block}'] = ResBlockLow(stage, block, filter, stride)
        self.net_layers['network_head'] = FC(units = architecture.get('num_class'))

    def call(self, X):
        x = self.net_layers['network_stem'](X)
        for stage, (filter, stride, blocks) in enumerate(zip(self.architecture.get('filters'),
                                                            self.architecture.get('strides'),
                                                            self.architecture.get('blocks'))):
            for block in range(blocks):
                x = self.net_layers[f'{stage}-{block}'](x)
                print(f'{stage+2}-{block+1}: {tf.shape(x)}')

        x = tf.math.reduce_mean(x, axis = [1, 2])
        x = self.net_layers['network_head'](x)

        return x

if __name__ == '__main__':
    import numpy as np
    architecture20 = {'filters': [16, 32, 64],
                    'blocks': [3, 3, 3],
                    'strides': [1, 2, 2],
                    'num_class': 10}

    model_low = ResNetLow(architecture20)
    model_low(np.random.randn(2, 32, 32, 3))
