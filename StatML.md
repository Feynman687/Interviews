Hey,

This list will cover the statistical ML topics that might be useful to anyone who's prepping for ML software positions. The depth in which one needs to study depends on multiple factors including the company, team in the company and the person him/herself.

* [Tree Based ML Algorithms](https://www.analyticsvidhya.com/blog/2016/04/tree-based-algorithms-complete-tutorial-scratch-in-python/)
 
    * What is entropy? Information gain (IG) concepts
    * [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) 
    * Bagging 
    * XGBoost (Why popular - parallelization)
    * Trees for classification versus regression
    * CART/Regression Trees, algorithmic change to incorporate regression in trees (maximum, mean of samples in each leaf to make final prediction)
    * Variance reduction method instead of IG

* Estimation strategies: Maximum likelihood (MLE) versus Maximum apriori (MAP)

* Naive Bayes, Logistic Regression
    * Generative versus Discriminative models
    * Logistic regression intuition from a perceptron
    * Loss functions for Logistic regression
    * Multiclass LR (derivations for likelihood estimation and gradient calculations)
    * How Multiclass LR is different from MLPs (Multi-layer perceptron)

* Regularization
    * Types, differences, uniqueness in norms L0, L1, L2 
    * Why L3, L4, L5, .. norms are not used
    * [Why is L1 sparse?](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)
    * Bagging - Boosting - Cross validation
    * Boosting loss similarity to log-loss/Logistic regression

* Regularization in Deep Networks
    * Dropouts
    * [BatchNorm](https://www.quora.com/Is-there-a-theory-for-why-batch-normalization-has-a-regularizing-effect) (Is it a regularizer?)
    * Data augmention as regularization
    * [Early stopping, multitask learning, adversarial learning](https://towardsdatascience.com/regularization-techniques-for-neural-networks-e55f295f2866)
    * [Zoneouts, dropconnect](https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b) (specifically for LSTMs)

* [PCA and SVD](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
    * [What PCA?](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
    * [Loss of PCA](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/)
    * Difference between the two, convexity of both their losses
    * Eigenvalue calculations
        * What they depict, why important
    
* Class imbalance issues
    * Algorithmic ways
        * [Weighing samples as per their effect on total loss - focal loss](https://medium.com/analytics-vidhya/how-focal-loss-fixes-the-class-imbalance-problem-in-object-detection-3d2e1c4da8d7)
    * Sampling ways 

* BayesNet and unsupervised learning
    * [Why inference on BayesNet is intractable?](https://www.quora.com/Why-is-exact-inference-in-a-bayesian-network-intractable)
    * Inference
        * Monte carlo methods
        * Giibs Sampling
    * [Expectation-Minimization](http://cs229.stanford.edu/notes/cs229-notes7b.pdf)
        * [More Math on EM and use in Hidden markov models](http://www.cs.cmu.edu/~aarti/Class/10701/readings/gentle_tut_HMM.pdf)
    * Gaussian Mixture models
    * KMeans - loss and [code from scratch](http://www.datasciencecourse.org/notes/unsupervised/)
    * KNNSs and how they are different from KMeans

* Metrics to test a model
    * Precision, recall, F1 - differences, use cases
    * AUC, area under ROC curve
    * What the area signifies? use-case based questions

* SVMs
    * Hinge loss
    * Code implementation

* Linear Regression - loss function calculation and derivations
    * MLE vs MAP (Different estimation strategies)
    * [How MAP brings regularization in linear regression loss](https://math.stackexchange.com/questions/2917109/map-solution-for-linear-regression-what-is-a-gaussian-prior)
    * [Convexity](https://stats.stackexchange.com/questions/160179/do-we-need-gradient-descent-to-find-the-coefficients-of-a-linear-regression-mode/164164%23164164), solving the loss directly 
    * Kernel regression

* ICA (Independent component analysis) - difference from PCA/SVD
    * When to use ICA?

* Difference in decision boundaries for all algorihtms (Tree vs Logistic vs Linear Reg vs SVMs vs Naive Bayes)









    