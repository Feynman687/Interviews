* MLPs forward and backprop, pytorch style code in python
* Softmax backprop (if ground truth "y" is one-hot, what's the simplified form for Cross-entropy)
* [Different Losses](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
    * Which loss to prefer when
    * KL Divergence (when true probability _p_ is not ground truth, entropy reduction of _p_), how is it different than xent

* [Log-linear model](https://en.wikipedia.org/wiki/Log-linear_model) and maximum entropy similarity
* Backpropagation in multiplicative networks
* Weight Initialization
    * Types, when to prefer what
* [Activation functions](https://blog.paperspace.com/vanishing-gradients-activation-function/)
    * ReLU and its types; what ReLU function evolution helps with in modeling a neural network
    * Other general functions

* [Story of optimizers](http://deeplearning.cs.cmu.edu/S20/document/slides/lec6.stochastic_gradient.pdf)
    * [How it all started?](https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/)
    * SGD
    * Newton's method
        * Why not incorporated, the main problem
        * Quasi-Newton methods
            * Limited method BFGS - used for log-linear models and CRFs
    * Momentum
    * [Nestorov Momentum](https://cs231n.github.io/neural-networks-3/#sgd)
    * Adagrad, RMS, Adam

* Residual networks
    * Why they work - [in brief](http://slazebni.cs.illinois.edu/spring17/lec04_advanced_cnn.pdf), [follow-up](https://arxiv.org/pdf/1603.05027.pdf)
        * The identity mapping helps in the flow of gradients back to initial layers, solving the weights being small issue due to vanishing gradients
        * Creates ensemble of networks and therefore removes layers which do not affect the resnet's final output
        * Gradient flow
        * Misnomer on vanishing gradient: VG due to input saturation is handled by batch norm
* Dense net, difference from resnets

* Sampling
    * [Sampling a number from its PDF](http://probcomp.csail.mit.edu/blog/programming-and-probability-sampling-from-a-discrete-distribution-over-an-infinite-set/)
    * Inverse Transform Sampling
    * Floyd's algo to generate _k_ of _m_ numbers
    * Rejection Sampling
    * Importance Sampling [intro](https://www.youtube.com/watch?v=V8f8ueBc9sY)
    * MCMC methods
    * Gibbs
    * Reservoir Sampling
    * Prioritized experience replay 

* CNNs
    * Usage, difference from MLPs, translation invariance
    * Backprop with code
    * Pooling, batching
    * Different architectures: Unet (upsampling); VGGs 
    * Difference from RNNs

* HMMs
    * Forward backward
    * Training and inference

* Viterbi decoding
    * Space time complexities

* [CRFs](https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture12.pdf)
    * Training and inference
    * MEMMs and how they relate to Logistic Regression

* Autoencoders
    * Differences with PCA

* RNNs vs LSTMs (Could be old due to xmers)
    * Why LSTMs better
    * Exploding vs Vanishing gradients
    * Regularization
        * Zoneouts
        * Randomized length BPTT
        * Activation Regularizatoin (AR)
        * Temporal AR
    * BPTT - what is it, how to code it

* Seq2seq models 
    * [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
    * [Attention isn't all you need](https://medium.com/synapse-dev/understanding-bert-transformer-attention-isnt-all-you-need-5839ebd396db)

* VAE, GANs
    * Basics (No interviewer ever went in-depth)

* Xmers
    * What, where use, why?
    * [BERT](https://arxiv.org/pdf/1810.04805.pdf), a xmer

* Tree based decoding over Attention models
* Transfer learning, one-shot learning
