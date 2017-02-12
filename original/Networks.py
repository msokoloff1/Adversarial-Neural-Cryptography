import tensorflow as tf
import Utilities as utils

class Network():
    def __init__(self, messageLength, name):
        """ Acts as an abstract class to improve code reusability"""
        self._inputMessage = tf.placeholder(tf.float32, [None,messageLength])
        self.name = name

    def _convLayer1D(self, input, numOutputChannels, filterWidth, stride, name, pad = 'SAME',activation = tf.nn.sigmoid, bias = False):
        """ Creates a layer that applies a one dimensional convolutional filter to the input"""
        with tf.variable_scope(name) as scope:
            input = utils.ensureRank3(input)
            numInputChannels = int(input.get_shape()[-1])
            filter = self._weightVar((filterWidth, numInputChannels , numOutputChannels ) )
            conv = tf.nn.conv1d(input, filter, stride = stride, padding = pad)
            if (bias):
                conv = conv + self._bias(numOutputChannels)
            return activation(conv)

    def _weightVar(self, shape):
        """ Helper function for generating a weight tensor"""
        weights = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    def _fcLayer(self, input, numOutputs, name, bias = False, activation = tf.nn.sigmoid):
        """ Creates a fully connected neural network layer"""
        with tf.variable_scope(name) as scope:
            input = utils.ensureRank2(input)
            shape1 = int(input.get_shape()[-1])
            weights = self._weightVar((int(shape1), int(numOutputs)))
            result = tf.matmul(input, weights)
            if(bias):
                result = result + self._bias(numOutputs)
            return activation(result)

    def _bias(self, shape):
        """ Helper function for generating a bias tensor"""
        bias = tf.constant(0.1, shape=shape)
        return tf.Variable(bias)

    def getInputTensor(self):
        """Safely returns input placeholder for reference"""
        return self._inputMessage

    def _combineKeyAndText(self, key, messageLength):
        """Concatenates text(ciphar or plain) with the key"""
        concatenated = tf.concat(1,(self._inputKey, utils.ensureRank2(self._inputMessage)))
        return concatenated

    def getUpdateOp(self, loss, optimizer):
        """Returns a tensor that, when executed, applies the gradients to the correct network"""
        clipNorm = 100 #<- not used now
        networkParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        grads = tf.gradients(loss, networkParams)
        grads, _ = tf.clip_by_global_norm(grads, clipNorm)
        self.apply_grads = optimizer.apply_gradients(zip(grads, networkParams))
        return self.apply_grads

class Encoder(Network):
    def __init__(self, messageLength, name):
        """Used to encode a message that cannot be understood by anyone except the desired recipient"""
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
            self.output = (conv4 + 1) /2.0

class Decoder(Network):
    def __init__(self, messageLength, alice,  name):
        """Used to decode a message that can only be understood by this network"""
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
            self.output = (conv4 + 1) /2.0

class UnauthDecoder(Network):
    def __init__(self, messageLength, alice, name):
        """Attempts to decrypt the ciphartext without authorization"""
        super().__init__(messageLength, name)
        print("Eve Instantiated")
        self._inputMessage = utils.ensureRank2(alice.output)
        with tf.variable_scope(name) as scope:
            fc1 = self._fcLayer(self._inputMessage, messageLength*2, 'e_fc1')
            conv1 = self._convLayer1D(fc1, numOutputChannels=2, filterWidth=4, stride=1, name='e_conv1')
            conv2 = self._convLayer1D(conv1, numOutputChannels=4, filterWidth=2, stride=2, name='e_conv2')
            conv3 = self._convLayer1D(conv2, numOutputChannels=4, filterWidth=1, stride=1, name='e_conv3')
            conv4 = self._convLayer1D(conv3, numOutputChannels=1, filterWidth=1, stride=1, name='e_conv4', activation = tf.nn.tanh)
            self.output = (conv4 + 1) /2.0

