import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import sys
from net.transkconv import TransK_Conv, AddCoords

class Conv_block(tf.keras.Model):
    def __init__(self, filters, imsize, name=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = TransK_Conv(filters=filters, kernel_size=kernel_size, padding='same', activation=None, name='conv0')
        #self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)

        self.coord1 = AddCoords(imsize=imsize)
        self.conv1 = TransK_Conv(filters=filters, kernel_size=kernel_size, padding='same', activation=None, name='conv1' )
        #self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        self.batch1 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=None):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        x0 = self.coord1(x0)
        x0 = self.conv1(x0, training=training)
        x0 = self.batch1(x0, training=training)
        x0 = tf.nn.relu(x0)
        return x0

    
class Conv_block_b1(tf.keras.Model):
    def __init__(self, filters, imsize, strides=[1,1], name=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block_b1, self).__init__(name=name)
        self.coord0 = AddCoords(imsize=imsize)
        self.conv0 = TransK_Conv(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=None,
                                       name='conv0')
        self.batch0 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=None):
        x0 = inputs
        x0 = self.coord0(x0)
        x0 = self.conv0(x0, training=training)
        x0 = self.batch0(x0, training=training)
        x0 = tf.nn.relu(x0)
        return x0


class Unet_Encode(tf.keras.Model):
    def __init__(self, classes, name=None):
        super(Unet_Encode, self).__init__(name=name)
        filters = 64
        self.conv0 = Conv_block(filters, imsize=512, block_nums=2)
        self.pool0 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same')
        self.conv1 = Conv_block(filters * 2, imsize=256, block_nums=2)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same')
        self.conv2 = Conv_block(filters * 4, imsize=128, block_nums=2)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same')
        self.conv3 = Conv_block(filters * 8, imsize=64, block_nums=2)

    def call(self, inputs, training=None):
        x0 = inputs
        x0 = self.conv0(x0, training=training)
        x1 = 1 * x0
        x0 = self.pool0(x0)
        x0 = self.conv1(x0, training=training)
        x2 = 1 * x0
        x0 = self.pool1(x0)
        x0 = self.conv2(x0, training=training)
        x3 = 1 * x0
        x0 = self.pool2(x0)
        x0 = self.conv3(x0, training=training)
        x4 = 1*x0 
        return x1, x2, x3, x4


class Unet_Decode(tf.keras.Model):
    def __init__(self, classes, name=None):
        super(Unet_Decode, self).__init__(name=name)
        filters = 64
        self.upsampling1 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate1 = tf.keras.layers.Concatenate(axis=-1)
        self.convu1 = Conv_block(filters * 4,imsize=128, block_nums=2)
        self.upsampling2 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate2 = tf.keras.layers.Concatenate(axis=-1)
        self.convu2 = Conv_block(filters * 2,imsize=256, block_nums=2)
        self.upsampling3 = tf.keras.layers.UpSampling2D(size=[2, 2], interpolation='bilinear')
        self.concatenate3 = tf.keras.layers.Concatenate(axis=-1)
        self.convu3 = Conv_block(filters,imsize=512, block_nums=2, kernel_size=[3, 3])

    def call(self, inputs, training=None):
        x1, x2, x3, x4 = inputs
        x0 = self.upsampling1(x4)
        x0 = self.concatenate1([x0, x3])
        x0 = self.convu1(x0, training=training)
        x0 = self.upsampling2(x0)
        x0 = self.concatenate2([x0, x2])
        x0 = self.convu2(x0, training=training)
        x0 = self.upsampling3(x0)
        x0 = self.concatenate3([x0, x1])
        outputs = self.convu3(x0, training=training)

        return outputs


class ConnectUnet(tf.keras.Model):
    def __init__(self,unetn=3):
        super(ConnectUnet, self).__init__()
        self.unetn=unetn
        self.unet_decode0 = Unet_Decode(3, name='unet_decode')
        self.unet_encode0 = Unet_Encode(3, name='unet_encode')
        self.conv_out0 = tf.keras.layers.Conv2D(filters=3, kernel_size=[1, 1], padding='same', activation=None)
        
        if unetn>1:
            self.unet_decode1 = Unet_Decode(3, name='unet_decode')
            self.unet_encode1 = Unet_Encode(3, name='unet_encode')
            self.conv_out1 = tf.keras.layers.Conv2D(filters=3, kernel_size=[1, 1], padding='same', activation=None)
        if unetn>2:
            self.unet_decode2 = Unet_Decode(3, name='unet_decode')
            self.unet_encode2 = Unet_Encode(3, name='unet_encode')
            self.conv_out2 = tf.keras.layers.Conv2D(filters=3, kernel_size=[1, 1], padding='same', activation=None)

    def call(self, inputs, training=None):
        inputs = inputs
        x0 = self.unet_encode0(inputs, training=training)
        x0 = self.unet_decode0(x0, training=training)
        output_seg0 = self.conv_out0(x0)
        output_seg0 = tf.nn.sigmoid(output_seg0)
        if self.unetn>1:
            x0 = tf.concat([x0,inputs],axis=-1)
            x0 = self.unet_encode1(x0, training=training)
            x0 = self.unet_decode1(x0, training=training)
            output_seg1 = self.conv_out1(x0)
            output_seg1 = tf.nn.sigmoid(output_seg1)
        if self.unetn>2:
            x0 = tf.concat([x0,inputs],axis=-1)
            x0 = self.unet_encode2(x0, training=training)
            x0 = self.unet_decode2(x0, training=training)
            output_seg2 = self.conv_out2(x0)
            output_seg2 = tf.nn.sigmoid(output_seg2)
        if self.unetn==3:
            return [output_seg0, output_seg1, output_seg2], x0
        elif self.unetn==2:
            return [output_seg0, output_seg1], x0
        elif self.unetn==1:
            return [output_seg0], x0




