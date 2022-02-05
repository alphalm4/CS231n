from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    
    # Modified
    delta = 1.0 # delta setting btw scores
    margin_bool = np.zeros((num_train, num_classes)) # to store nonzero margin idx of classes

    for i in range(num_train):
        scores = X[i].dot(W) # (1,D) dot (D,C)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                margin_bool[i][j] = True # to store nonzero margin idx of classes

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss. (L2 REG)
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW_data = np.zeros(W.shape)
    dscores = np.zeros(W.shape[1]) # Grad of score vector (1, C)

    # Gradient Contribution of One Data Point X[i].dot(W)
    
    for i in range(num_train):
        for j in range(num_classes):
            if j == y[i] :
                dscores[j] = np.sum(margin_bool[i]) * (-1/num_train)
            elif margin_bool[i][j] == True :
                dscores[j] = 1/num_train
            else:
                dscores[j] = 0
        
        
        dW_data = np.dot(X[i].reshape(-1,1), dscores.reshape(1,-1))
        dW += dW_data

    # Gradient Contribution of the L2 Regularization Term
    
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1.0 # delta SVM setting btw scores

    scores = np.dot(X,W) # (N, C)
    answer_scores = scores[np.arange(num_train), y] # np.arange(N) : (N, ), y : (N, )
    margins = np.maximum(scores - answer_scores.reshape(-1,1) + delta, 0) # reshape to broadcast sum
    margins[np.arange(num_train), y] = 0 # for answer entries

    loss += np.mean(np.sum(margins, axis=1)) + 2*reg*np.sum(W*W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dmargins : 1/N for entries below ↓
    #            1) margins > 0      2) (i,j) is not (i, y[i])

    dscores = np.zeros(scores.shape) # (N, C)
    dscores += (margins > 0) * 1.0
    dscores[np.arange(num_train), y] -= np.sum(margins > 0, axis = 1)
    dscores = dscores / num_train
    # array coordinate를 [np.array , np.array]로 받음

    dW += np.dot(X.T, dscores) # (D, N) dot (N, C)
    dW += 2*reg*W
      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
