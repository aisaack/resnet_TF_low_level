import tensorflow as tf

class Conv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = 'same',
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in'), # 'he normal'
        kernel_regularizer = None,
        use_bias = True,
        bias_initializer = tf.keras.initializers.Zeros(),
        bias_regularizer = None,
        activation = None,
        trainable = True,
        name = 'conv',
        **kwargs
    ):
        super(Conv2d, self).__init__(name=name, **kwargs)
        self.filters = filters
        if isinstance(kernel_size, tuple):
            self.kernel_size = list(kernel_size)
        elif isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 2

        if isinstance(strides, tuple):
            self.strides = list(strides)
        elif isinstance(strides, int):
            self.strides = [strides] * 2

        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.activation = activation
        self.trainable = trainable

    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = self.kernel_size + [input_shape[-1], self.filters],
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      trainable = self.trainable,)
        if self.use_bias:
            self.bias = self.add_weight(name = 'bias',
                                        shape = (self.filters,),
                                        initializer = self.bias_initializer,
                                        regularizer = self.bias_regularizer,
                                        trainable = self.trainable)

    def call(self, X):
        conv = tf.nn.convolution(X,
                                 filters = self.kernel,
                                 strides = self.strides,
                                 padding = self.padding.upper(),
                                 data_format = 'NHWC'
                                 )
        if self.use_bias:
            conv = tf.nn.bias_add(conv, self.bias)

        if self.activation:
            conv = self.activation(conv)

        return conv


class FC(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation = None,
        weight_initializer = tf.keras.initializers.VarianceScaling(scale=2, mode='fan_in'),
        weight_regularizer = None,
        use_bias = True,
        bias_initializer = tf.keras.initializers.Zeros(),
        bias_regularizer = None,
        trainable = True,        
        name = 'FC',
        **kwargs
    ):
        super(FC, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.trainable = trainable
        
    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.weight = self.add_weight(name = 'weight',
                                      shape = [input_shape[-1], self.units],
                                      initializer = self.weight_initializer,
                                      regularizer = None,
                                      trainable = self.trainable)
        if self.use_bias:
            self.bias = self.add_weight(name = 'bias',
                                        shape = (self.units,),
                                        initializer = self.bias_initializer,
                                        regularizer = None,
                                        trainable = self.trainable)
            
    def call(self, X):
        fc = tf.linalg.matmul(X, self.weight)

        if self.use_bias:
            fc = tf.nn.bias_add(fc, self.bias)
        
        if self.activation:
            fc = self.activation(fc)

        return fc


class BatchNorm(tf.keras.layers.Layer):
    def __init__(
      self,      
      gamma = 'ones',
      beta = 'zeros', 
      moving_mean = 'zeros',
      moving_var = 'ones',
      momentum = 0.99,
      epsilon = 1e-6,
      center = True,
      scale = True,
      activation = None,
      trainable = True,
      name = 'batch_norm',
      **kwargs  
    ):
        super(BatchNorm, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.beta = beta 
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.activation = activation
        self.trainable = trainable

    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name = 'moving_mean',
                                           shape = [1] * (len(input_shape)-1) + [input_shape[-1]],
                                           initializer = tf.keras.initializers.get(self.moving_mean),
                                           trainable = False)
        
        self.moving_var = self.add_weight(name = 'moving_var',
                                          shape = [1] * (len(input_shape)-1) + [input_shape[-1]],
                                          initializer = tf.keras.initializers.get(self.moving_var),
                                          trainable = False)
        
        if self.scale:
            self.gamma = self.add_weight(name = 'gamma',
                                        shape = [1] * (len(input_shape)-1) + [input_shape[-1]],
                                        initializer = tf.keras.initializers.get(self.gamma),
                                        trainable = self.trainable)
        else:
            self.gamma = 0
        
        if self.center:
            self.beta = self.add_weight(name = 'beta',
                                        shape = [1] * (len(input_shape)-1) + [input_shape[-1]],
                                        initializer = tf.keras.initializers.get(self.beta),
                                        trainable = self.trainable)
        else:
            self.beta = 1

    def call(self, X, training = None):
        if training:
            mean, var = tf.nn.moments(X,
                                      axis = list(range(len(X.shape) - 1)),
                                      keepdims = True)
            
            updated_mean = tf.math.add(tf.math.multiply(self.moving_mean, self.momentum),
                                       tf.math.multiply(mean, tf.math.subtract(1, self.momentum)))
            self.moving_mean.assgin(updated_mean)
            
            updated_var = tf.math.add(tf.math.multiply(self.moving_var, self.momentum),
                                      tf.math.multiply(var, tf.math.subtract(1, self.momentum)))
            self.moving_var.assgin(updated_var)
        else:
            mean = self.moving_mean
            var = self.moving_var
        
        gamma = self.gamma
        beta = self.beta

        bn = tf.nn.batch_normalization(X,
                                       mean,
                                       var,
                                       offset = beta,
                                       scale = gamma,
                                       variance_epsilon = self.epsilon)
        
        if self.activation:
            bn = self.activation(bn)        

        return bn
