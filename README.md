# Machine Learning, Data Science and Deep Learning

Exercises are related to Udemy course instructed by Frank Kane (Amazon) from Sundog Education.

## Tools

- Python
- matplotlib
- numpy
- pandas
- seaborn
- anaconda
- Jupyter Notebook
- Spyder IDE
- Apache Spark
- TensorFlow

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

### Recommendation Systems

#### User-based collaborative filtering

1. Compute similarity scores between users
2. recommend stuff similar users liked and you haven’t seen yet

Con: Peple’s taste changes, more people than things

#### Item-based collaborative filtering

1. Find every pair of e.g. movies that were watched by same person
2. Measure similarity of ratings across all users who watched both
3. Sort by movie by simlarity

Pro: always same thing, fewer things than people => faster to compute

### Data mining techniques

- Binning: numeric data into categorical, e.g. age => age range
- Transforming: logarithmic transform
- Encoding: buckets for each category
- Scaling/Normalization: some models work only when valuess normally distributed around 0
- Shuffling: Shuffle training data

## Big data

-

### Data Warehousing

- ELT: Extract-Load-Transform (processing happens in a tool, raw data directly loaded into system)
- ETL: Extract-Transform-Load (preprocessing raw data before loading, old school way)

## Scripts

Open jupyter notebook (make sure you are in the folder where .ipynb file is):

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
