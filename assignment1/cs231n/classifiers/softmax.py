from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    dim, num_classes = W.shape

    for i in range(num_train):
        # Update loss
        scores = np.dot(X[i], W) # (D, ) , (D, C)
        exp_scores = np.exp(scores)
        prob = exp_scores[y[i]] / np.sum(exp_scores)
        loss_it = -np.log(prob)
        loss += loss_it

        # Update dW
        dexp = (-1) * np.ones(10) * prob / exp_scores[y[i]]
        dexp[y[i]] += 1/exp_scores[y[i]]
        dexp = dexp / (-1 * num_train)
        
        dscores = exp_scores * dexp # (C, )
        dW += np.dot( X[i].reshape(-1,1), dscores.reshape(1,-1) ) # (D,C)
    
    loss /= num_train
    loss += reg*np.sum(W*W)

    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    dim, num_classes = W.shape

    # Update Loss
    scores = np.dot(X,W) # (N, C)
    exp_scores = np.exp(scores)
    prob = exp_scores[ np.arange(num_train), y] / np.sum(exp_scores, axis=1) # (N, )
    loss += np.mean( (-1) * np.log(prob) )
    loss += reg * np.sum(W*W)

    # Update dW
    dprob = np.zeros_like(scores) # (N, C)
    dprob += (prob / exp_scores[np.arange(num_train), y]).reshape(num_train, -1) # Broadcast Sum
    dprob[np.arange(num_train), y] += (-1) / exp_scores[np.arange(num_train), y] # modify the answer entries
    dprob /= num_train
    dscores = dprob * exp_scores # elementwise
    dW += np.dot(X.T, dscores)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
