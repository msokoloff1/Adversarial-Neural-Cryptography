import argparse
import tensorflow as tf
import Networks as nets
import Utilities as utils
import numpy as np

"""Parse command line args"""
parser = argparse.ArgumentParser()
parser.add_argument('-num_iters'     , default=10000, type=int, help="Sets the number of training iterations")
parser.add_argument('-message_length', default=16   , type=int, help="Length of plaintext/ciphertext")
parser.add_argument('-batch_size'    , default=4096  , type=int, help="Batch size used for training ops")
parser.add_argument('-optimizer'     , default='Adam', type=str, help="Optimizer to be used when applying gradients (adam,adadelta,adagrad,rmsprop)")
parser.add_argument('-learning_rate' , default=0.0008, type=int, help="Learning rate to be used when applying gradients")
parser.add_argument('-variantNet' , action= 'store_true', help="Use the variant nets")
parser.add_argument('-tbPrefix', default = "_", type = str, help= "Prefix for tensorboard, if running multiple experiments")
args = parser.parse_args()

"""Select optimizer"""
optimizer = tf.train.AdamOptimizer(args.learning_rate)
if(args.optimizer.lower() == 'adadelta'):
    optimizer = tf.train.AdadeltaOptimizer(args.learning_rate)

elif(args.optimizer.lower() == 'adagrad'):
    optimizer = tf.train.AdagradOptimizer(args.learning_rate)

elif(args.optimizer.lower() == 'rmsprop'):
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)

"""Instantiate nets"""
if(args.variantNet):
    print("Variant")
    alice = nets.EncoderVariant(args.message_length, 'aliceNet')
    bob = nets.DecoderVariant(args.message_length, alice, 'aliceNet')
    eve = nets.UnauthDecoderVariant(args.message_length, alice, 'aliceNet')
else:
    print("Og Nets")
    alice = nets.Encoder(args.message_length , 'aliceNet')
    bob   = nets.Decoder(args.message_length   , alice ,'aliceNet')
    eve   = nets.UnauthDecoder(args.message_length   , alice ,'aliceNet')

"""Calculate loss metrics"""
aliceAndBobLoss =utils.getBobAliceLoss(bob, eve, alice, args.message_length)
eveLoss = utils.getEveLoss(eve, alice)

"""Create generator for obtaining the correct update op"""
turnGen = utils.getTurn(  alice.getUpdateOp(aliceAndBobLoss, optimizer)
                        , bob.getUpdateOp(aliceAndBobLoss, optimizer)
                        , eve.getUpdateOp(eveLoss,optimizer)
                        )

"""Instantiate logger, provide command line args for context"""
#logger = utils.log(details = args._get_kwargs())



"""Begin training loop"""
def train(numIters):
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        fileWriter = tf.summary.FileWriter(args.prefix+'tensorboard', sess.graph)

        dataGen    = utils.getData(args.message_length, args.batch_size)
        #logMetrics = utils.getLoggingMetrics(bob, eve, alice)
        sess.run(tf.global_variables_initializer())
        
        for iter in range(args.num_iters):
            data = next(dataGen)
            feedDict = {
                  alice._inputKey     : np.array(data['key'])
                , alice._inputMessage : np.array(data['plainText'])
            }
            updateOps = next(turnGen)
            sess.run(updateOps, feed_dict=feedDict)
            if(iter%100 == 0):
                summary, aliceAndBobLossEvaluated,eveLossEvaluated =  sess.run([merged, tf.reduce_mean(aliceAndBobLoss),tf.reduce_mean(eveLoss)], feed_dict=feedDict)
                fileWriter.add_summary(summary, iter)


train(args.num_iters)






