import tensorflow as tf
import sys

class AddCoords(tf.keras.Model):
    def __init__(self, imsize, with_r=True, xoy=2):
        super(AddCoords, self).__init__()
        self.xoy = xoy
        self.with_r = with_r
        self.x_dim = imsize
        self.y_dim = imsize
            

    def call(self, input_tensor):
        self.batch_size_tensor, _, __ = input_tensor.shape[:3]
        xx_channel = tf.range(self.y_dim, dtype=tf.float32)
        xcs = tf.math.sin(xx_channel / tf.math.pow(10000., tf.cast(xx_channel // 2, dtype=tf.float32) * 2 / self.y_dim))
        xcc = tf.math.cos(xx_channel / tf.math.pow(10000., tf.cast(xx_channel // 2, dtype=tf.float32) * 2 / self.y_dim))
        xx_channel = tf.where(xx_channel % 2 == 0, xcs, xcc)
        xx_channel = tf.expand_dims(xx_channel, 0)
        xx_channel = tf.expand_dims(xx_channel, 0)  # shape [1,1,y_dim]
        xx_channel = tf.tile(xx_channel, [self.batch_size_tensor, self.x_dim, 1])
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_channel = tf.range(self.x_dim, dtype=tf.float32)
        ycs = tf.math.sin(yy_channel / tf.math.pow(10000., tf.cast(yy_channel // 2, dtype=tf.float32) * 2 / self.x_dim))
        ycc = tf.math.cos(yy_channel / tf.math.pow(10000., tf.cast(yy_channel // 2, dtype=tf.float32) * 2 / self.x_dim))
        yy_channel = tf.where(yy_channel % 2 == 0, ycs, ycc)
        yy_channel = tf.expand_dims(yy_channel, 0)
        yy_channel = tf.expand_dims(yy_channel, -1)  # shape [1,x_dim, 1]
        yy_channel = tf.tile(yy_channel, [self.batch_size_tensor, 1, self.y_dim])
        yy_channel = tf.expand_dims(yy_channel, -1)

        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = tf.math.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
            ret = tf.concat([ret, rr], axis=-1)

        return ret




class AttentionBlock(tf.keras.layers.Layer):

    def __init__(self, filters, codename, heads):
        super(AttentionBlock, self).__init__()
        self.filters = filters
        self.codename = codename
        self.heads = heads

    def build(self, input_shape):
        input_shape0 = input_shape[0]
        self.new_shape = [input_shape0[0], input_shape0[1], self.heads, self.filters]
        self.out_shape = [input_shape0[0], input_shape0[1], self.filters * self.heads]

    def call(self, inputs):
        qw, kw, vw = inputs
        dk0 = tf.math.sqrt(tf.cast(self.filters, dtype = tf.float32))
        en_q = qw
        en_q = tf.reshape(en_q, self.new_shape)
        en_q = tf.transpose(en_q, perm=[2, 0, 1, 3])

        en_k = kw
        en_k = tf.reshape(en_k, self.new_shape)
        en_k = tf.transpose(en_k, perm=[2, 0, 1, 3])

        en_v = vw
        en_v = tf.reshape(en_v, self.new_shape)
        en_v = tf.transpose(en_v, perm=[2, 0, 1, 3])

        en_a = tf.nn.softmax(tf.einsum('h b i j, h b k j -> h b i k', en_q, en_k) / dk0)
        en_f = tf.einsum('h b i j, h b j k -> h b i k', en_a, en_v)
        en_f = tf.transpose(en_f, perm=[1, 2, 0, 3])
        return en_f


class MultiAttentionBlock(tf.keras.layers.Layer):

    def __init__(self, filters, codename, heads=1):
        super(MultiAttentionBlock, self).__init__()
        self.filters = filters
        self.codename = codename
        self.heads = heads

    def build(self, input_shape):
        input_shape0 = input_shape[0]
        self.outchannel_shape = self.filters
        self.encode_dense_q = tf.keras.layers.Dense(self.filters * self.heads, name=self.codename + '_q')
        self.encode_dense_k = tf.keras.layers.Dense(self.filters * self.heads, name=self.codename + '_k')
        self.encode_dense_v = tf.keras.layers.Dense(self.filters * self.heads, name=self.codename + '_v')
        self.out_shape = [input_shape0[0], input_shape0[1], self.outchannel_shape * self.heads]
        self.new_shape = [input_shape0[0], input_shape0[1], self.heads, self.filters]
        self.attentions = AttentionBlock(self.filters, self.codename, self.heads)
        self.dense0 = tf.keras.layers.Dense(1000, name='dense')
        self.dense1 = tf.keras.layers.Dense(self.filters, name='dense')
        self.batch0 = tf.keras.layers.GroupNormalization(groups=self.outchannel_shape * self.heads, center=False,
                                                         scale=False)
        self.batch1 = tf.keras.layers.GroupNormalization(groups=self.outchannel_shape, center=False, scale=False)

    def call(self, inputs, training=True):
        qw, kw, vw = inputs
        qw = self.encode_dense_q(qw)
        kw = self.encode_dense_k(kw)
        vw = self.encode_dense_v(vw)
        en_f = self.attentions([qw, kw, vw])
        en_f = en_f + tf.reshape(qw, self.new_shape)
        en_f = tf.reshape(en_f, self.out_shape)
        en_f0 = self.batch0(en_f, training=training)
        en_f = self.dense1(en_f0)
        en_f = en_f + tf.reduce_sum(tf.reshape(en_f0, self.new_shape), axis=-2)
        en_f = self.batch1(en_f, training=True)

        return en_f


class TransK_Conv(tf.keras.layers.Layer):

    def __init__(self, filters, name=None, kernel_size=[3, 3], strides=1, padding='same', activation=None, use_bias=True,
                 if_top=False):
        super(TransK_Conv, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.upper()
        self.activation = activation
        self.use_bias = use_bias
        self.if_top = not if_top
        self.strides = strides

    def build(self, input_shape):
        self.kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.filters, self.filters)
        self.w0 = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True, name='weight'
        )
        self.new_shape = (self.kernel_size[0] * self.kernel_size[1], input_shape[-1], self.filters)
        if self.use_bias:
            b_init = tf.zeros_initializer()
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer="zeros",
                trainable=True, name='bias'
            )
        self.dense0 = tf.keras.layers.Dense(self.filters, name='dense')
        self.encode_attention = MultiAttentionBlock(self.filters, codename='encode')
        self.decode_attention = MultiAttentionBlock(self.filters, codename='decode')
        self.fb_attention = MultiAttentionBlock(self.filters, codename='fb')

    def call(self, inputs, training=True):
        x0 = inputs
        w = self.w0

        if self.use_bias:
            output0 = tf.nn.conv2d(x0, w, self.strides, padding=self.padding) + self.b
        else:
            output0 = tf.nn.conv2d(x0, w, self.strides, padding=self.padding)
        w_v = tf.reshape(w, self.new_shape)
        w_v1 = tf.transpose(w_v, perm=[1, 0, 2])
        en_f = self.encode_attention([w_v1, w_v1, w_v1], training=training)
        en_f = tf.transpose(en_f, perm=[2, 1, 0])
        w_v0 = tf.transpose(w_v, perm=[2, 0, 1])
        fb_f = self.fb_attention([w_v0, w_v0, w_v0], training=training)
        de_f = self.decode_attention([en_f, fb_f, fb_f], training=training)

        de_f = tf.transpose(de_f, perm=[1, 2, 0])
        de_f = tf.reshape(de_f, self.kernel_shape)
        de_f = self.dense0(de_f)

        output1 = tf.nn.conv2d(output0, de_f, [1,1], padding=self.padding)

        return tf.concat([output0, output1], axis=-1)

class Conv_block(tf.keras.Model):
    def __init__(self, filters, kernel_size, imsize, strides=[1,1], padding='same', use_bias=True, name=None):
        super(Conv_block, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = TransK_Conv(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        return x0
    
class Conv_block_no(tf.keras.Model):
    def __init__(self, filters, kernel_size, imsize, strides=[1,1], padding='same', use_bias=True, name=None):
        super(Conv_block_no, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = TransK_Conv(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        return x0