from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        C_in, H_in, W_in = input_dim
        F_conv = num_filters
        H_conv = filter_size
        W_conv = filter_size
        D_aff  = hidden_dim
        C_out  = num_classes
        weight_scale = weight_scale
        
        # initialize : conv - relu - 2x2 maxpool
        # conv_param  and pool_param will be initialized in loss function
        # dimension flow : (N, C_in, H_in, W_in) → (N, F_conv, H_in, W_in)
        # since here W and H are preserved
        # otherwise, H_out = 1 + (H_in + 2*pad - H_conv) / stride_conv
        layer_num = 1
        W_str = 'W' + str(layer_num)
        b_str = 'b' + str(layer_num)

        self.params[W_str] = np.random.normal(0.0, weight_scale, size=(F_conv, C_in, H_conv, W_conv))
        self.params[b_str] = np.zeros(F_conv)
        
        # initialize : affine - relu
        # dimension flow : (N, F_conv, H_in/2, W_in/2) = (N, F_conv * (H_in/2) * (W_in/2)) → (N, D_aff)

        layer_num += 1
        W_str = 'W' + str(layer_num)
        b_str = 'b' + str(layer_num)
        
        self.params[W_str] = np.random.normal(0.0, weight_scale, size=(int(F_conv*(H_in/2)*(W_in/2)), D_aff))
        self.params[b_str] = np.zeros(D_aff)

        # initialize : affine - softmax
        # dimension flow : (N, D_aff) → (N, C_out)
        layer_num += 1
        W_str = 'W' + str(layer_num)
        b_str = 'b' + str(layer_num)

        self.params[W_str] = np.random.normal(0.0, weight_scale, size=(D_aff, C_out))
        self.params[b_str] = np.zeros(C_out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        # SIX CACHES

        X = X.astype(self.dtype)
        self.caches = []

        out, cache = conv_forward_fast(X, W1, b1, conv_param)
        self.caches.append(cache)
        out, cache = relu_forward(out)
        self.caches.append(cache)
        out, cache = max_pool_forward_fast(out, pool_param)
        self.caches.append(cache)
        
        out, cache = affine_forward(out, W2, b2)
        self.caches.append(cache)
        out, cache = relu_forward(out)
        self.caches.append(cache)
        
        out, cache = affine_forward(out, W3, b3)
        self.caches.append(cache)
        
        scores = out
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        for i in range(3) :
          W_str = 'W' + str(i+1)
          loss += 0.5 * self.reg * np.sum(self.params[W_str]**2)
        
        W_str = 'W' + str(3)
        b_str = 'b' + str(3)
        dout, grads[W_str], grads[b_str] = affine_backward(dscores, self.caches[5])

        W_str = 'W' + str(2)
        b_str = 'b' + str(2)
        dout = relu_backward(dout, self.caches[4])
        dout, grads[W_str], grads[b_str] = affine_backward(dout, self.caches[3])

        W_str = 'W' + str(1)
        b_str = 'b' + str(1)
        dout = max_pool_backward_fast(dout, self.caches[2])
        dout = relu_backward(dout, self.caches[1])
        dout, grads[W_str], grads[b_str] = conv_backward_fast(dout, self.caches[0])

        for i in range(3) :
          W_str = 'W' + str(i+1)
          grads[W_str] += self.reg * self.params[W_str]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
