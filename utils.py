import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten
from keras.layers.pooling import _GlobalPooling2D


#Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-5,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    def build(self, input_shape):    
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        content, style_mean, style_std = inputs[0], inputs[1], inputs[2]
        input_shape = K.int_shape(content)
        reduction_axes = [1, 2]  #list(range(0, len(input_shape)))

        mean = K.mean(content, reduction_axes, keepdims=True)
        stddev = K.std(content, reduction_axes, keepdims=True) + self.epsilon
#        styled_content = style_mean + style_std * (content - mean) / stddev
        styled_content = content * style_std
        return styled_content
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        content_shape = input_shape[0]
        return content_shape

class AdaIN(Layer):
    """ Borrowed and modified https://github.com/liuwei16/adain-keras/blob/master/layers.py """
    def __init__(self, data_format='channels_last', eps=1e-5, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.spatial_axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        self.eps = eps

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        """  inputs = [content, style_mean, style_std]  """
        content, style_mean, style_std = inputs
        content_mean, content_var = tf.nn.moments(content, self.spatial_axis, keep_dims=True)
        out = tf.nn.batch_normalization(content,
                                        content_mean,
                                        content_var,
                                        style_mean,
                                        style_std,
                                        self.eps)
        return out

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#def adain(x, y_mean, y_scale, y_bias, epsilon=1e-5):
#    '''
#    Borrowed from https://github.com/jonrei/tf-AdaIN
#    Normalizes the `content_features` with scaling and offset from `style_features`.
#    See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
#    '''
#    x_mean, x_variance = tf.nn.moments(x, [1, 2], keep_dims=True)
#    normalized_x = tf.nn.batch_normalization(
#        x,
#        x_mean,
#        x_variance,
#        y_mean,
#        y_scale,
#        epsilon
#        )
#    return normalized_content_features


class DenseSN(Dense):
    """ Borrowed from https://github.com/IShengFang/SpectralNormalizationKeras """
    def __init__(self, units, **kwargs):
        super(DenseSN, self).__init__(units, **kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super(DenseSN, self).build(input_shape)
        

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-5):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 

class ConvSN2D(Conv2D):
    """ Borrowed from https://github.com/IShengFang/SpectralNormalizationKeras """
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvSN2D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
#        self.built = True
        super(ConvSN2D, self).build(input_shape)

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-5):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])



from keras.engine.topology import Layer
from keras import initializers

class Bias(Layer):
    """
    Custom keras layer that simply adds a scalar bias to each location in the input
    """
    
    def __init__(self, initializer='zeros', **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.initializer = initializer
    
    def build(self, input_shape):
        self.bias = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[-1],), initializer=self.initializer)
        super(Bias, self).build(input_shape)
    
    def call(self, x):
        return K.bias_add(x, self.bias, data_format='channels_last')
#        return x + self.bias



# Self attention is Taken from https://gist.github.com/sthalles/507ce723226274db8097c24c5359d88a
class SelfAttention(Layer):
    def __init__(self, number_of_filters, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.number_of_filters = number_of_filters
        
        self.f = ConvSN2D(number_of_filters // 8, 1, strides=1, padding='same', name="f_x")
        self.g = ConvSN2D(number_of_filters // 8, 1, strides=1, padding='same', name="g_x")
        self.h = ConvSN2D(number_of_filters,    1, strides=1, padding='same', name="h_x")
        self.gamma = tf.contrib.eager.Variable(0., trainable=True, name="gamma")
#        self.flatten = tf.keras.layers.Flatten()
        self.flatten = Flatten()

    def hw_flatten(self, x):
        # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
        # flat the feature volume across the width and height dimensions 
        x_shape = tf.shape(x)
        return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]
    
    def call(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = self.hw_flatten(f)
        g_flatten = self.hw_flatten(g)
        h_flatten = self.hw_flatten(h)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True) # [B,N,C] * [B, N, C] = [B, N, N]

        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x
        return y

    def get_config(self):
        config = {'number_of_filters': self.number_of_filters}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
