import tensorflow as tf
from net.resnet50 import ResNet50
from net.resnet18 import ResNet18

class AddCoords(tf.keras.Model):
    """
    Add Coord to tensor
    Alternate implementation, use tf.tile instead of tf.matmul, and x_dim, y_dim is got directly from input_tensor
    """

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


class Full_Conv(tf.keras.layers.Layer):

    def __init__(self, filters, name=None, kernel_size=[3, 3], strides=[1,1], padding='same', activation=None, use_bias=True,
                 if_top=False):
        super(Full_Conv, self).__init__(name=name)
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
    def __init__(self, filters, imsize, name=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = Full_Conv(filters=filters, kernel_size=kernel_size, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        #self.act0 = tf.keras.layers.PReLU()

        self.coord1 = AddCoords(imsize=imsize)
        self.conv1 = Full_Conv(filters=filters, kernel_size=kernel_size, padding='same', activation=None, name='conv1' )
        #self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        #self.act1 = tf.keras.layers.PReLU()

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        #x0 = tf.nn.gelu(x0)
        x0 = self.coord1(x0)
        x0 = self.conv1(x0, training=training)
        x0 = self.batch1(x0, training=training)
        x0 = tf.nn.relu(x0)
        #x0 = tf.nn.gelu(x0)
        return x0

class Conv_block_u1(tf.keras.Model):
    def __init__(self, filters, imsize, name=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block_u1, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = Full_Conv(filters=filters, kernel_size=kernel_size, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        #self.act0 = tf.keras.layers.PReLU()

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        #x0 = tf.nn.gelu(x0)
        return x0    
    
class Conv_block_b1(tf.keras.Model):
    def __init__(self, filters, imsize, strides=[1,1], name=None, kernel_size=[3, 3]):
        super(Conv_block_b1, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = Full_Conv(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.act0 = tf.keras.layers.PReLU()

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = self.act0(x0)
        return x0

class Dblock(tf.keras.Model):
    def __init__(self, filters, imsize, use_bias=True, name=None, block_nums=1, kernel_size=[4, 4], use_norm=True):
        super(Dblock, self).__init__(name=name)
        self.use_norm=use_norm
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = Full_Conv(filters=filters, kernel_size=kernel_size, strides=[2,2], padding='same', activation=None, use_bias=use_bias, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=[2,2], padding='same', activation=None, use_bias=use_bias)
        if use_norm:
            self.batch0 = tf.keras.layers.GroupNormalization(groups=-1)
        self.act0 = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs, training=True):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        if self.use_norm:
            x0 = self.batch0(x0, training=training)
        x0 = self.act0(x0)
        return x0

class Cenblock(tf.keras.Model):
    def __init__(self, name=None):
        super(Cenblock, self).__init__(name=name)
        filters = 64
        self.upsampling0_0 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv0_0 = Conv_block_b1(filters,imsize=512)

        self.upsampling1_0 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv1_0 = Conv_block_b1(filters,imsize=256)
        
        self.upsampling1_1 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv1_1 = Conv_block_b1(filters,imsize=512)

        self.upsampling2_0 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv2_0 = Conv_block_b1(filters,imsize=128)

        self.upsampling2_1 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv2_1 = Conv_block_b1(filters,imsize=256)
        
        self.upsampling2_2 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.conv2_2 = Conv_block_b1(filters,imsize=512)

        self.conv_out0 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.act0 = tf.keras.layers.PReLU()
        self.conv_out1 = tf.keras.layers.Conv2D(filters=9, kernel_size=[1, 1], padding='same', activation=None)

    def call(self, inputs, training=None):
        x1, x2, x3 = inputs
        x1 = self.upsampling1_0(x1)
        x1 = self.conv0_0(x1, training=training)

        x2 = self.upsampling1_0(x2)
        x2 = self.conv1_0(x2, training=training)
        
        x2 = self.upsampling1_1(x2)
        x2 = self.conv1_1(x2, training=training)

        x3 = self.upsampling2_0(x3)
        x3 = self.conv2_0(x3, training=training)

        x3 = self.upsampling2_1(x3)
        x3 = self.conv2_1(x3, training=training)
        
        x3 = self.upsampling2_2(x3)
        x3 = self.conv2_2(x3, training=training)

        outputs = self.conv_out0(tf.concat([x3, x2, x1], axis=-1))
        outputs = self.batch0(outputs)
        outputs = self.act0(outputs)
        outputs = self.conv_out1(outputs)

        return tf.nn.sigmoid(outputs)

class Unet_Decode(tf.keras.Model):
    def __init__(self, classes, name=None):
        super(Unet_Decode, self).__init__(name=name)
        filters = 64
        self.upsampling1 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate1 = tf.keras.layers.Concatenate(axis=-1)
        self.convu1 = Conv_block(filters * 4,imsize=64, block_nums=2)
        self.upsampling2 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate2 = tf.keras.layers.Concatenate(axis=-1)
        self.convu2 = Conv_block(filters * 2,imsize=128, block_nums=2)
        self.upsampling3 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate3 = tf.keras.layers.Concatenate(axis=-1)
        self.convu3 = Conv_block(filters,imsize=256, block_nums=2, kernel_size=[3, 3])
        self.upsampling4 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.convu4 = Conv_block(filters//2,imsize=512, block_nums=2, kernel_size=[3, 3])
        self.convuu1 = Conv_block_u1(filters * 4,imsize=64, block_nums=2)
        self.convuu2 = Conv_block_u1(filters * 2,imsize=128, block_nums=2)
        self.convuu3 = Conv_block_u1(filters * 1,imsize=256, block_nums=2)

    def call(self, inputs, training=None):
        x1, x2, x3, x4 = inputs
        x0 = self.upsampling1(x4)
        x3 = self.convuu1(x3, training=training)
        x0 = self.concatenate1([x0, x3])
        x0 = self.convu1(x0, training=training)
        x0 = self.upsampling2(x0)
        x2 = self.convuu2(x2, training=training)
        x0 = self.concatenate2([x0, x2])
        x0 = self.convu2(x0, training=training)
        x0 = self.upsampling3(x0)
        x1 = self.convuu3(x1, training=training)
        x0 = self.concatenate3([x0, x1])
        outputs = self.convu3(x0, training=training)
        outputs = self.upsampling4(outputs)
        outputs = self.convu4(outputs, training=training)

        return outputs

class GeneratorNet(tf.keras.Model):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        '''self.backbone_out_layers = ["conv1_relu", "conv2_block2_out", "conv3_block2_out", "conv4_block2_out"]
        base_model = ResNet18(input_shape=(512, 512, 3), include_top=False,)

        self.unet_encode0 = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer(x).output for x in self.backbone_out_layers])'''
        self.backbone_out_layers = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
        #self.backbone_out_layers = ["conv1_relu", "conv2_block2_out", "conv3_block2_out", "conv4_block2_out"]
        '''base_model = tf.keras.applications.ResNet50(input_shape=(512, 512, 3),
                                                          include_top=False,
                                                          weights='imagenet')'''
        base_model = ResNet50(input_shape=(512, 512, 3), include_top=False)

        self.unet_encode0 = tf.keras.Model(inputs=base_model.input,
                                       outputs=[base_model.get_layer(x).output for x in self.backbone_out_layers])
        self.unet_decode0 = Unet_Decode(3, name='unet_decode')
        self.conv_out0 = tf.keras.layers.Conv2D(filters=3, kernel_size=[1, 1], padding='same', activation=None)
        self.cenb0 = Cenblock()
        #self.batchout = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=None):
        inputs = inputs
        x0 = self.unet_encode0(inputs, training=training)
        out0 = self.cenb0(x0[:3], training=training)
        x0 = self.unet_decode0(x0, training=training)
        output_seg0 = self.conv_out0(x0)
        #output_seg0 = self.batchout(output_seg0, training=training)
        output_seg0 = tf.nn.sigmoid(output_seg0)
        return output_seg0, out0

class VGGNet(tf.keras.Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.backbone_out_layers = ["block1_conv2", "block2_conv2", "block3_conv2", "block3_conv4"]
        base_model = tf.keras.applications.VGG19(input_shape=(512, 512, 3), weights='imagenet', include_top=False,)
        #base_model = tf.keras.applications.VGG19(input_shape=(512, 512, 3), weights=None, include_top=False,)

        self.vgg19 = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer(x).output for x in self.backbone_out_layers])
        self.mean0 = tf.constant([[[[0.485, 0.406, 0.456]]]])
        self.std0 = tf.constant([[[[0.229, 0.225, 0.224]]]])

    def call(self, inputs, training=None):
        inputs = (inputs-self.mean0)/self.std0
        x0 = self.vgg19(inputs, training=training)
        return x0

class DescriminatorNet(tf.keras.Model):
    def __init__(self):
        super(DescriminatorNet, self).__init__()
        filters = 64
        self.conv1 = Dblock(filters * 1,imsize=512, use_norm=False)
        self.conv2 = Dblock(filters * 2,imsize=256, use_bias=False)
        self.conv3 = Dblock(filters * 4,imsize=128, use_bias=False)
        self.conv4 = Dblock(filters * 8,imsize=64, use_bias=False)
        self.conv5 = Dblock(filters * 16,imsize=32, use_bias=False)
        self.conv_out0 = tf.keras.layers.Conv2D(filters=2, kernel_size=[3, 3], padding='same', activation=None)

    def call(self, inputs, training=None):
        inputs = inputs
        x0 = self.conv1(inputs, training=training)
        x0 = self.conv2(x0, training=training)
        x0 = self.conv3(x0, training=training)
        x0 = self.conv4(x0, training=training)
        x0 = self.conv5(x0, training=training)
        x0 = self.conv_out0(x0)
        x0 = tf.nn.sigmoid(x0)
        return x0