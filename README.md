# Machine Learning, Data Science and Deep Learning

Exercises are related to Udemy course instructed by Frank Kane (Amazon) from Sundog Education.

## Tools

- Python
- matplotlib (dataviz)
- numpy (support for large multi-dimensional arrays and matrices)
- scipy (scientific computing and technical computing)
- pandas (data analysis and manipulation)
- seaborn (fancy dataviz)
- scikit learn (machine learning in Python)
- anaconda (Python Data Science Platform)
- Jupyter Notebook (create and share documents that contain live code)
- Spyder (IDE shipped inside Anaconda)
- Apache Spark (Tool for Big data)
- TensorFlow (Machine learning tools)
- Keras (part of Tensorflow to support neural networks out of the box, compatible with scikit-learn)

## Projects

- Prediction of car prices based on features (Regression)
- Movie recommendation system (Item-based Collaborative filtering)
- Email Spam Blocker (Naive Bayes)
- Wikipedia keyword search engine (TD-IDF)
- CV filterer (Decision tree)
- Taxi pickup and dropoff fastest route calculation (Q-Learning with Gym)
- Predict whether voter is Democrat or Republican based on data how voter has voted on different topics in the US (Keras, Sigmoid+Relu)
- Handwriting recognition (CNN, Keras)
- Predict whether user liked the movie based on review (RNN, Sentiment Analysis, Keras, IMDB data)
- Holiday Image Recognition (Keras ResNet)
- Supervised learning for belign/malignant cancer tumours using variety of different methods and comparing them together

## Topics

### Unsupervised Learning

Output results not available, finds patterns, does not predict results

- K-Means Clustering (groups data points into clusters)
- PCA (Principal Component Analysis, reduces dimensions)

### Supervised Learning

Predicts result based on existing data and results

- Train/Test split (scikit-learn)
- Naive Bayes Classifier
- Regression (Linear, Polynomial, multi-level)
- Decision trees (create if/else path to determine)
- KNN: K-nearest neighbours (predict result based on n neighbours)
- Support Vector Machines (SVM)

Metrics to measure classifiers:

- Precision (when you care about false positives, eg. drug test)
- recall (when you care about false negatives, eg. fraud detection)
- F1 score
- ROC: Receive Iperating Characteristic Curve
- AUC: Area Under Curve

### Reinforcement Learning

"Agent" explores space and learns value of different state changes in different conditions, e.g. Pacman. Mathematical framework called **_Markov decision process (MDP)_**. Used in **_Dynamic Programming_** to solve complex problems by breaking it into simpler subproblems.

#### Q-Learning

- set of states (s), set of actions (a), value of each state/action (Q)
- Q start value 0
- when exploring space:
  - good thing happens => increase Q
  - bad thing happens => decrease Q

### Ensemble Learning

Multiple models solve same problem and vote on results

- **_Bagging_** = multiple models trained by random subsets of data (bootstrap aggregating)
- **_Boosting_** = each model boosts attributes that address data mis-classified by previous model, eg. XGBoost

##### Recommendation Systems

### User-based collaborative filtering

1. Compute similarity scores between users
2. recommend stuff similar users liked and you haven’t seen yet

Con: Peple’s taste changes, more people than things

### Item-based collaborative filtering

1. Find every pair of e.g. movies that were watched by same person
2. Measure similarity of ratings across all users who watched both
3. Sort by movie by simlarity

Pro: always same thing, fewer things than people => faster to compute

### TF-IDF

Term Frequency and Inverse Document Frequency: Important data for search, figures out what terms are most relevant for a document. Assumes document is a "Bag of words", no relationships/typos/tenses/synonyms between words taken into account

- Term Frequency: how often word occurs in a document
- Document Frequency: how often word occurs in a set of documents, e.g. all wikipedia pages, all websites etc
- relevancy of a word to a document: Term Frequency / Document Frequency (How often a word appears in a document compared to appearance everywhere, avoids putting weight on words like a/the/and etc)
- in practice we use log of IDF since word counts are distributed exponentially => better weighting with log

### Data mining techniques

- Binning: numeric data into categorical, e.g. age => age range
- Transforming: logarithmic transform
- Encoding: buckets for each category
- Scaling/Normalization: some models work only when valuess normally distributed around 0
- Shuffling: Shuffle training data

### Big data

- RDD: Resilient distributed datasets RDD
  - Resilient: if cluster down it recovers itself, tries agai etc
  - Distributed: distribute computing to cluster of computers instead of single CPU
- Dataframe: dataset of row objects, dataset knows types ahead of time => faster

### Data Warehousing

- ELT: Extract-Load-Transform (processing happens in a tool, raw data directly loaded into system)
- ETL: Extract-Transform-Load (preprocessing raw data before loading, old school way)

### Statictical tools

- T-test: difference between datasets relative to the variance in the data (higher T => real difference), e.g. revenue conversion
- Fisher's exact test (e.g. clickthrough rates)
- E-test (e.g. transaction per user)
- chi-squared test (e.g. product quantities purchased)
- P-value: low P-value => there is a real statistically significant difference, e.g. < 0.005

## Deep Learning

- Gradient Descent (minimize error): Pick a random point and measure error until we found local minima of the error that is optimized, used to train neural network
- Autodiff (find gradients for gradient descent): technique to speed up gradient descent by calculating partial derivatives in number of outputs (few outputs, many inputs)
- Softmax (choose most probable classification): used for classification, produces probabilities for each class and highest probability is the answer, used e.g. face recognition
- Learning Rate: start at random point, sample solutions (weights) seeking to minimize const function over many epochs => How far apart these samples are is the learning rate (is hyperparameter). Large learning rate can overshoot correct answer, small learning rate increase training time
- Batch Size: how many samples within each epoch, smaller batch work their way out of local minima better than large ones => too large batch sizes can get stuck to local minima and give wrong solution
- in deep learning there are 3 types of dataset:
  - Training dataset: used to fit the data from the bottom up
  - Evaluation dataset: evaluates data each epoch against evaluation dataset
  - Testing dataset: used for fully trained model to test it
- Often computationally expensive

### Neural Networks

- **_Artificial Neuron_**: single unit in deep neural network, used with NOT/AND/OR operations that decide whether it will "fire" or not
- **_LTU_**: Linear Threshold Unit, adds weight to input
- **_Perceptron_**: layer of LTU's, create system that learns over time, cells that fire together wire together
- **_Multi-layer perceptron (MLP)_**: adds hidden layers, deep neural network
- **_A modern deep neural network_**: apply softmax to output, train using gradient descent
- **_Backpropagation_**: train MLP weights, gradient descent using reverse-mode autodiff
  1. compute output error
  2. compute how much each neuron in the previous hidden layer contributed
  3. Backpropagate that error in a reverse pass
  4. Tweak weights to reduce the error using gradient descent
- **_Rectifier_**: Activation function, step function, e.g. ReLU (Rectifier Linear Unit)
- **_Optimization function_**: faster optimizers than gradient descent, e.g. Momentum Optimization

### Regularization: Methods to avoid overfitting

- Sign when this is happening: high accuracy on training data, low on test and evaluation data
- use simpler model with less layers/neurons
- **_Dropout_**: ignore e.g. 50% of all neurons randomly while training is still ongoing, forces model to spread its learning, used often in CNN
- **_Early stopping_** when performance starts dropping

#### Convolutional Neural Networks (CNN)

- Break data into chunks (= convolutions) and process those individually (=subsampling) and feed into higher layers
- Find features that aren't in a specific spot (e.g. stop sign can be anywhere in the image)
- Process:
  1. Local receptive fields scan image looking for edges and pick up edges oof e.g. stop sign
  2. Edges sent to higher level convolutions that identify more details (shape, letters etc)
  3. shape matched against of pattern what stop sign looks like
- Used e.g. Machine translation, sentence claassification, Sentiment analysis
- Keras had Conv1D, Conv2D, Conv3D, typical usage: Conv2D => MaxPooling2D => dropout => flatten => dense => dropout => softmax
- Multiple libraries/architectures vailable such as AlexNet, ResNet (Residual network), GoogLeNet
- LSTM Cell: Long short-term memory cell, maintains separate short-term and long-term states to avoid dilution of states
- GRU Cell: Gated Recurrent Unit, simplified LSTM Cell

#### Recurrent Neural Network (RNN)

- Sequences in data e.g. time series (web logs, stock trades)
- Sequences of arbitrary length e.g. Machine Translation, Image Captions, Machine-generated music
- also feeds data back, not just passing through the neuron
- past behaviour of neuron influences its new behaviour in next sequence (aka memory cell)
- RNN Topologies:
  - Sequence to sequence: predict price based on series of historical data
  - Sequence to vector: words in a sentence to sentiment
  - Vector to sequence: create captions from an image
  - encoder => decoder: sequence => vector => sequence, machine translation

### Tensorflow

- Tool to optimize the processing of graph and distribute processing across network, can be run outside datacenters too, in phone for example
- tensor: name for an array or matrix of values
- Tensorflow steps:
  1. construct graph to compute tensors
  2. Initialize variables
  3. execute graph (nothing happens until then)
- Neural networks usually work best with normalized data (sklearn StandardScaler)
- Play around with neural networks at [http://playground.tensorflow.org/](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.11043&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

### Transfer Learning

- Reusing trained models as they are or tune them to applications
- You can find existing models from Model Zoos, e.g. Caffe Model Zoo

## Scripts

Open jupyter notebook (make sure you are in the folder where .ipynb file is, to run code inside noteboof press shift+enter):

```
jupyter notebook
```

Run spark script

```
spark-submit SparkDecisionTree.py
```

Open anaconda's own IDE Spyder

```
spyder
```

Set up an Anaconda environment for tensorflow (supports currently only Python 3.7)

```
conda create -n py37 tensorflow python=3.7
```

Run jupyter with Tensorflow compatible env

```
conda activate py37
conda install jupyter
jupyter notebook
```

Exit Python 3.7 environment and get back to 3.8

```
conda deactivate
```
