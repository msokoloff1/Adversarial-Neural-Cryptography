import argparse
import tensorflow as tf
import Networks as nets
import Utilities as utils
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-num_iters'     , default=100000, type=int, help="Sets the number of training iterations")
parser.add_argument('-message_length', default=32    , type=int, help="Length of plaintext/ciphertext")
parser.add_argument('-batch_size'    , default=4096  , type=int, help="Batch size used for training ops")
parser.add_argument('-optimizer'     , default='Adam', type=str, help="Optimizer to be used when applying gradients (adam,adadelta,adagrad,rmsprop)")
parser.add_argument('-learning_rate' , default=0.0008, type=int, help="Learning rate to be used when applying gradients")
args = parser.parse_args()

#Select optimizer
optimizer = tf.train.AdamOptimizer(args.learning_rate)
if(args.optimizer.lower() == 'adadelta'):
    optimizer = tf.train.AdadeltaOptimizer(args.learning_rate)

elif(args.optimizer.lower() == 'adagrad'):
    optimizer = tf.train.AdagradOptimizer(args.learning_rate)

elif(args.optimizer.lower() == 'rmsprop'):
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)

#Instantiate nets
alice = nets.Alice(args.message_length , 'aliceNet')
bob   = nets.Bob(args.message_length   , alice ,'aliceNet')
eve   = nets.Eve(args.message_length   , alice ,'aliceNet')

#Calculate loss metrics
aliceAndBobLoss =utils.getBobAliceLoss(bob, eve, alice, args.message_length)
eveLoss = utils.getEveLoss(eve, alice)

#Create generator for obtaining the correct update op
turnGen = utils.getTurn(  alice.getUpdateOp(aliceAndBobLoss, optimizer)
                        , bob.getUpdateOp(aliceAndBobLoss, optimizer)
                        , eve.getUpdateOp(eveLoss,optimizer)
                        )



def train(numIters):
    with tf.Session() as sess:
        dataGen    = utils.getData(args.message_length, args.batch_size)
        logMetrics = utils.getLoggingMetrics(bob, eve, alice)
        sess.run(tf.initialize_all_variables())
        for iter in range(args.num_iters):

            data = next(dataGen)

            feedDict = {
                  alice._inputKey     : np.array(data['key'])
                , alice._inputMessage : np.array(data['plainText'])
            }

            updateOps = next(turnGen)
            sess.run(updateOps, feed_dict=feedDict)

            if(iter%100 == 0):
                aliceAndBobLossEvaluated,eveLossEvaluated,eveIncorrect, bobIncorrect  =  sess.run([tf.reduce_mean(aliceAndBobLoss),tf.reduce_mean(eveLoss)] + logMetrics, feed_dict=feedDict)
                print("Iteration %s | Alice/Bob Loss : %g | Eve Loss : %g | Eve Incorrect : %g | Bob Incorrect : %g"%(str(iter).zfill(6),aliceAndBobLossEvaluated, eveLossEvaluated,eveIncorrect,bobIncorrect))


train(args.num_iters)






