#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


# def sigmoid(x):
#     """
#     Compute the sigmoid function for the input here.
#     Arguments:
#     x -- A scalar or numpy array.
#     Return:
#     s -- sigmoid(x)
#     """

#     ### YOUR CODE HERE
#     s = 1/(1+np.exp(-x))

#     ### END YOUR CODE

#     return s


# def naiveSoftmaxLossAndGradient(
#     centerWordVec,
#     outsideWordIdx,
#     outsideVectors,
#     dataset
# ):
#     """ Naive Softmax loss & gradient function for word2vec models

#     Implement the naive softmax loss and gradients between a center word's 
#     embedding and an outside word's embedding. This will be the building block
#     for our word2vec models.

#     Arguments:
#     centerWordVec -- numpy ndarray, center word's embedding
#                     (v_c in the pdf handout)
#     outsideWordIdx -- integer, the index of the outside word
#                     (o of u_o in the pdf handout)
#     outsideVectors -- outside vectors (rows of matrix) for all words in vocab
#                       (U in the pdf handout)
#     dataset -- needed for negative sampling, unused here.

#     Return:
#     loss -- naive softmax loss
#     gradCenterVec -- the gradient with respect to the center word vector
#                      (dJ / dv_c in the pdf handout)
#     gradOutsideVecs -- the gradient with respect to all the outside word vectors
#                     (dJ / dU)
#     """

#     ### YOUR CODE HERE

#     ### Please use the provided softmax function (imported earlier in this file)
#     ### This numerically stable implementation helps you avoid issues pertaining
#     ### to integer overflow. 
#     naive_softmax = softmax(np.dot(outsideVectors, centerWordVec))
#     loss = -np.log(naive_softmax[outsideWordIdx])

#    # compute gradient with respect to center word vector
#     gradCenterVec = -outsideVectors[outsideWordIdx] + np.dot(naive_softmax, outsideVectors)

#    # compute gradient with respect to outside word vectors
#     gradOutsideVecs = np.dot(naive_softmax.reshape(-1, 1), centerWordVec.reshape(1, -1)) # when w != o
#     gradOutsideVecs[outsideWordIdx] = np.dot((naive_softmax[outsideWordIdx].reshape(-1, 1) - 1.0), centerWordVec.reshape(1, -1))

  
    
       


#     ### END YOUR CODE

#     return loss, gradCenterVec, gradOutsideVecs


# def getNegativeSamples(outsideWordIdx, dataset, K):
#     """ Samples K indexes which are not the outsideWordIdx """

#     negSampleWordIndices = [None] * K
#     for k in range(K):
#         newidx = dataset.sampleTokenIdx()
#         while newidx == outsideWordIdx:
#             newidx = dataset.sampleTokenIdx()
#         negSampleWordIndices[k] = newidx
#     return negSampleWordIndices


# def negSamplingLossAndGradient(
#     centerWordVec,
#     outsideWordIdx,
#     outsideVectors,
#     dataset,
#     K=10
# ):
#     """ Negative sampling loss function for word2vec models

#     Implement the negative sampling loss and gradients for a centerWordVec
#     and a outsideWordIdx word vector as a building block for word2vec
#     models. K is the number of negative samples to take.

#     Note: The same word may be negatively sampled multiple times. For
#     example if an outside word is sampled twice, you shall have to
#     double count the gradient with respect to this word. Thrice if
#     it was sampled three times, and so forth.

#     Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
#     """

#     # Negative sampling of words is done for you. Do not modify this if you
#     # wish to match the autograder and receive points!
#     negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
#     indices = [outsideWordIdx] + negSampleWordIndices

#     ### YOUR CODE HERE

#     ### Please use your implementation of sigmoid in here.
#     u_o = outsideVectors[outsideWordIdx]
    
#     u_k = outsideVectors[negSampleWordIndices]

#     score_o = np.dot(u_o,centerWordVec)
#     score_k = np.dot(u_k,centerWordVec)

#     sig_o = sigmoid(score_o)
#     sig_k = sigmoid(-score_k)

#     loss = -np.log(sig_o)-np.sum(np.log(sig_k))

#     gradCenterVec = (sig_o-1)*u_o + np.sum((1-sig_k)[:,np.newaxis]*u_k,axis=0)

#     gradOutsideVecs = np.zeros_like(outsideVectors)

#     gradOutsideVecs[outsideWordIdx]+= (sig_o -1)*centerWordVec

#     for i,neg_idx in enumerate(negSampleWordIndices):
#         gradOutsideVecs[neg_idx]+=(1-sig_k[i])*centerWordVec

#     ### END YOUR CODE

#     return loss, gradCenterVec, gradOutsideVecs


# def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
#              centerWordVectors, outsideVectors, dataset,
#              word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
#     """ Skip-gram model in word2vec

#     Implement the skip-gram model in this function.

#     Arguments:
#     currentCenterWord -- a string of the current center word
#     windowSize -- integer, context window size
#     outsideWords -- list of no more than 2*windowSize strings, the outside words
#     word2Ind -- a dictionary that maps words to their indices in
#               the word vector list
#     centerWordVectors -- center word vectors (as rows) for all words in vocab
#                         (V in pdf handout)
#     outsideVectors -- outside word vectors (as rows) for all words in vocab
#                     (U in pdf handout)
#     word2vecLossAndGradient -- the loss and gradient function for
#                                a prediction vector given the outsideWordIdx
#                                word vectors, could be one of the two
#                                loss functions you implemented above.

#     Return:
#     loss -- the loss function value for the skip-gram model
#             (J in the pdf handout)
#     gradCenterVecs -- the gradient with respect to the center word vectors
#             (dJ / dV in the pdf handout)
#     gradOutsideVectors -- the gradient with respect to the outside word vectors
#                         (dJ / dU in the pdf handout)
#     """

#     loss = 0.0
#     gradCenterVecs = np.zeros(centerWordVectors.shape)
#     gradOutsideVectors = np.zeros(outsideVectors.shape)

#     ### YOUR CODE HERE
      

#     for ow in outsideWords:
#        outsideWordIdx = word2Ind[ow]
#        centerWordIdx = word2Ind[currentCenterWord]
#        centerWordVec = centerWordVectors[centerWordIdx]
#        loss_j,gradCenterVec_j,gradOutsideVecs_j = word2vecLossAndGradient(centerWordVec,outsideWordIdx,outsideVectors,dataset)
#        loss += loss_j
#        gradCenterVecs[centerWordIdx]+=gradCenterVec_j
#        gradOutsideVectors+=gradCenterVec_j

#     ### END YOUR CODE

#     return loss, gradCenterVecs, gradOutsideVectors

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    
    gradOutsideVecs = np.zeros_like(outsideVectors)
    # obtain y_hat (i.e., the conditional probability distribution p(O = o | C = c))
    # by taking vector dot products and applying softmax
    y_hat = softmax(np.dot(outsideVectors, centerWordVec)) # (N,) N x 1
    # can also get y_hat in a single line: y_hat = softmax(outsideVectors @ centerWordVec)

    # for a single pair of words c and o, the loss is given by:
    # J(v_c, o, U) = -log P(O = o | C = c) = -log [y_hat[o]]
    loss = -np.log(y_hat[outsideWordIdx])

    # grad calc
    # generate the ground-truth one-hot vector, [..., 0, outsideWordIdx=1, 0, ...]
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1
    # can also get loss as -np.dot(y, np.log(y_hat))    
    
    gradCenterVec = np.dot(y_hat - y, outsideVectors) # inner product results in a scalar
    # or gradCenterVec = np.dot(outsideVectors.T, y_hat - y)
    
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec) # outer product results in a matrix
    # or gradOutsideVecs = np.dot((y_hat - y)[:, np.newaxis], centerWordVec[np.newaxis, :]) 
    
    # sanity check the dimensions
    assert gradCenterVec.shape == centerWordVec.shape
    assert gradOutsideVecs.shape == outsideVectors.shape  

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.
    
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    
    # Calculate the first term
    y_hat = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))
    loss = -np.log(y_hat)
    
    gradCenterVec = np.dot(y_hat - 1, outsideVectors[outsideWordIdx])
    gradOutsideVecs[outsideWordIdx] = np.dot(y_hat - 1, centerWordVec)

    # Calculate the second term
    for i in range(K):
        w_k = indices[i+1]
        y_k_hat = sigmoid(-np.dot(outsideVectors[w_k], centerWordVec))
        loss += -np.log(y_k_hat)
        gradOutsideVecs[w_k] += np.dot(1.0 - y_k_hat, centerWordVec)
        gradCenterVec += np.dot(1.0 - y_k_hat, outsideVectors[w_k])

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    
    # skip-gram model predicts outside words from the center word
    
    # get center word vec first from currentCenterWord
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]

    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        stepLoss, gradCenter, gradOutside = word2vecLossAndGradient(centerWordVec,
                                                                    outsideWordIdx,
                                                                    outsideVectors,
                                                                    dataset)

        loss += stepLoss
        gradCenterVecs[centerWordIdx] += gradCenter
        gradOutsideVectors += gradOutside    

    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors
#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
