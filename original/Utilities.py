import tensorflow as tf
import threading
import numpy as np
import time
import datetime
import os

class log():
    def __init__(self, details, fileName = 'log.txt'):
        info   = '|'.join(["%s-%s"%(str(x[0]),str(x[1])) for x in details])
        header = ','.join(['Iteration','Alice-Bob/Loss','EveLoss', 'EveIncorrect', 'Alice/Bob-Incorrect', 'RunNumber'])
        self.time   = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H_%M_%S')
        self.index = len(os.listdir('logs'))
        with open('logs/log' + self.time +'.txt', 'a') as file:
            file.write(info +  '\n')
            file.write(header + '\n')
              
    def writeToFile(self, listOfParams):
        assert len(listOfParams) == 5, "Parameter list has too many/missing elements"
        with open('logs/log' + self.time + '.txt', 'a') as file:
            file.write(','.join([str(x) for x in listOfParams]) + ','+str(self.index) +'\n')

def ensureRank2(input):
    """Ensures the input tensor has a rank of 2, otherwise it reshapes the tensor"""
    if (len(input.get_shape()) == 3):
        input = tf.reshape(input, ([-1, int(input.get_shape()[1])]))
    tf.assert_rank(input, 2, message="Tensor is not rank 2")
    return input

def ensureRank3(input):
    """Ensures the input tensor has a rank of 3, otherwise it reshapes the tensor"""
    if (len(input.get_shape()) == 2):
        input = tf.expand_dims(input, 2)
    tf.assert_rank(input, 3, message="Tensor is not rank 3")
    return input

def getBobAliceLoss(bob, eve, alice, messageLength):
    """Calculates the loss for bob and alice and is returned as a tensor"""
    bobOutput = ensureRank2(bob.output)
    eveOutput = ensureRank2(eve.output)
    conversationLoss = tf.abs(bobOutput - alice._inputMessage)
    eveLoss   = tf.reduce_mean(tf.abs(eveOutput - alice._inputMessage))
    snoopLoss = tf.reduce_mean(tf.div(tf.pow( ((messageLength/2)-eveLoss),2), tf.pow((messageLength/2),2)))
    totalLoss = conversationLoss + 7 * snoopLoss
    return totalLoss

def getLoggingMetrics(bob, eve, alice):
    """Calculates the accuracy for bob and eve and it is returned as a tensor for logging"""
    with tf.name_scope('performance'):
        eveOutput = ensureRank2(eve.output)
        answer    = ensureRank2(alice._inputMessage)
        bobOutput = ensureRank2(bob.output)
        eveIncorrect = tf.reduce_mean(tf.abs(eveOutput - answer))
        bobIncorrect = tf.reduce_mean(tf.abs(bobOutput - answer))
        tf.summary.scalar('eve-Incorrect', eveIncorrect)
        tf.summary.scalar('bob/alice-Incorrect', bobIncorrect)
    return [eveIncorrect, bobIncorrect]

def getEveLoss(eve, alice):
    """Calculates the accuracy of eve and it is returned as a tensor"""
    eveOutput = ensureRank2(eve.output)
    answer    = ensureRank2(alice._inputMessage)
    eveLoss = tf.abs(eveOutput - answer)
    return eveLoss

def getTurn(aliceUpdateOp, bobUpdateOp, eveUpdateOp):
    """Generates a list containing the appropriate update op"""
    while True:
        yield [aliceUpdateOp, bobUpdateOp]
        yield [eveUpdateOp] #<- update eve twice in a row
        yield [eveUpdateOp]

def getData(numBits, batchSize):
    """Generates random data for training"""
    queue = []
    getBitSequence = lambda numBits, BatchSize: queue.insert(0, np.random.randint(2, size=(batchSize, numBits)))
    for _ in range(100) : getBitSequence(numBits, batchSize)
    while True:
        for _ in range(2) : threading.Thread(target=getBitSequence, args=(numBits,batchSize,)).start()
        yield {'key' : queue.pop(), 'plainText' : queue.pop()}
        


