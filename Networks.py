import tensorflow as tf
import Utilities as utils

class Network():
    def __init__(self, messageLength, name):
        self._inputMessage = tf.placeholder(tf.float32, [None,messageLength])
        self.name = name

    def _convLayer1D(self, input, numOutputChannels, filterWidth, stride, name, pad = 'SAME',activation = tf.nn.sigmoid, bias = False):
        with tf.variable_scope(name) as scope:
            input = utils.ensureRank3(input)
            numInputChannels = int(input.get_shape()[-1])
            filter = self._weightVar((filterWidth, numInputChannels , numOutputChannels ) )
            conv = tf.nn.conv1d(input, filter, stride = stride, padding = pad)

            if (bias):
                conv = conv + self._bias(numOutputChannels)



            return activation(conv)

    def _weightVar(self, shape):
        weights = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    def _fcLayer(self, input, numOutputs, name, bias = False, activation = tf.nn.sigmoid):
        with tf.variable_scope(name) as scope:

            input = utils.ensureRank2(input)

            shape1 = int(input.get_shape()[-1])
            weights = self._weightVar((int(shape1), int(numOutputs)))
            result = tf.matmul(input, weights)

            if(bias):
                result = result + self._bias(numOutputs)

            return activation(result)

    def _bias(self, shape):
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias)

    def getInputTensor(self):
        return self._inputMessage

    def _combineKeyAndText(self, key, messageLength):
            concatenated = tf.concat(1,(self._inputKey, utils.ensureRank2(self._inputMessage)))
            #Support for fully connected first layer:
            #indys = [,indexer(x)] for x in range(messageLength)
            #                                         for indexer in (lambda y: y, lambda y: y + messageLength)]
            #combinedInput = tf.gather_nd(concatenated, indys)
            return concatenated#combinedInput

    def getUpdateOp(self, loss, optimizer):
        clipNorm = 100 #<- not used now
        networkParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        grads = tf.gradients(loss, networkParams)
        grads, _ = tf.clip_by_global_norm(grads, clipNorm)
        self.apply_grads = optimizer.apply_gradients(zip(grads, networkParams))
        return self.apply_grads


class Alice(Network):
    def __init__(self, messageLength, name):
        super().__init__(messageLength,name)
        print("Alice Instantiated")
        self._inputKey = tf.placeholder(tf.float32, [None, messageLength], name ="alicePH")
        combinedInput = self._combineKeyAndText(self._inputKey, messageLength)
        with tf.variable_scope(name) as scope:
            fc1 = self._fcLayer(combinedInput, messageLength * 2, 'a_fc1')
            conv1 = self._convLayer1D(fc1,   numOutputChannels=2, filterWidth=4, stride=1, name='a_conv1')
            conv2 = self._convLayer1D(conv1, numOutputChannels=4, filterWidth=2, stride=2, name='a_conv2')
            conv3 = self._convLayer1D(conv2, numOutputChannels=4, filterWidth=1, stride=1, name='a_conv3')
            conv4 = self._convLayer1D(conv3, numOutputChannels=1, filterWidth=1, stride=1, name='a_conv4', activation = tf.nn.tanh)
            self.output = conv4


class Bob(Network):
    def __init__(self, messageLength, alice,  name):
        super().__init__(messageLength,name)
        print("Bob Instantiated")
        self._inputMessage = alice.output
        self._inputKey = alice._inputKey
        combinedInput = self._combineKeyAndText(self._inputKey, messageLength)
        with tf.variable_scope(name) as scope:
            fc1 = self._fcLayer(combinedInput, messageLength * 2,'b_fc1')
            conv1 = self._convLayer1D(fc1,   numOutputChannels=2, filterWidth=4, stride=1, name='b_conv1')
            conv2 = self._convLayer1D(conv1, numOutputChannels=4, filterWidth=2, stride=2, name='b_conv2')
            conv3 = self._convLayer1D(conv2, numOutputChannels=4, filterWidth=1, stride=1, name='b_conv3')
            conv4 = self._convLayer1D(conv3, numOutputChannels=1, filterWidth=1, stride=1, name='b_conv4', activation = tf.nn.tanh)
            self.output = conv4

class Eve(Network):
    def __init__(self, messageLength, alice, name):
        super().__init__(messageLength, name)
        print("Eve Instantiated")
        self._inputMessage = utils.ensureRank2(alice.output)
        with tf.variable_scope(name) as scope:
            fc1 = self._fcLayer(self._inputMessage, messageLength*2, 'e_fc1')
            conv1 = self._convLayer1D(fc1, numOutputChannels=2, filterWidth=4, stride=1, name='e_conv1')
            conv2 = self._convLayer1D(conv1, numOutputChannels=4, filterWidth=2, stride=2, name='e_conv2')
            conv3 = self._convLayer1D(conv2, numOutputChannels=4, filterWidth=1, stride=1, name='e_conv3')
            conv4 = self._convLayer1D(conv3, numOutputChannels=1, filterWidth=1, stride=1, name='e_conv4', activation = tf.nn.tanh)

            self.output = conv4

