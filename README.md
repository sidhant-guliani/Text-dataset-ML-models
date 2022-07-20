# Text-dataset-ML-models

In this repository we are analysing text data using `Traditional models` like: Naive Bays, SVM, logistic regression and using tpretrained transformer: BERT model.

### Preprocessing:

The first step in the development of any NLP model is text preprocessing. This means we're going to transform our texts from word sequences to feature vectors. These feature vectors contain their values for each of a large number of features.To obtain the weighted feature vectors, we combine the `CountVectorizer` and `TfidfTransformer` in a `Pipeline`, and fit this pipeline on the training data. We then transform both the training texts and the test texts to a collection of such weighted feature vectors. Scikit-learn also has a `TfidfVectorizer`, which achieves the same result as our pipeline.

## Training

Next, we train a text classifier on the preprocessed training data. We're going to experiment with three classic text classification models: Naive Bayes, Support Vector Machines and Logistic Regression. 

[Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are extremely simple classifiers that assume all features are independent of each other. They just learn how frequent all classes are and how frequently each feature occurs in a class. To classify a new text, they simply multiply the probabilities for every feature $x_i$ given each class $C$ and pick the class that gives the highest probability: 

\begin{equation*}
\hat y = argmax_k\  p(C_k) \prod_{i=1}^n p(x_i \mid C_k)
\end{equation*}

Naive Bayes Classifiers are very quick to train, but usually fall behind in terms of performance.

[Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) are much more advanced than Naive Bayes classifiers. They try to find the hyperplane in the feature space that best separates the data from the different classes. They do so by picking the hyperplane that maximizes the distance to the nearest data point on each side. When the classes are not linearly separable, SVMs map the data into a higher-dimensional space where a linear separation can hopefully be found. SVMs often achieve very good performance in text classification tasks.

[Logistic Regression models](https://en.wikipedia.org/wiki/Logistic_regression), finally, model the log-odds $l$, or $log(p/(1-p))$, of a class as a linear model and estimate the parameters $\beta$ of the model during training: 

\begin{equation*}
l = \beta_0 + \sum_{i=1}^n \beta_i x_i
\end{equation*}

Like SVMs, they often achieve great performance in text classification.

## Extensive evaluation

### Detailed scores

So far we've only looked at the accuracy of our models: the proportion of test examples for which their prediction is correct. This is fine as a first evaluation, but it doesn't give us much insight in what mistakes the models make and why. We'll therefore perform a much more extensive evaluation, in three steps. Let's start by computing the precision, recall and F-score of the best SVM for the individual classes:

- Precision is the number of times the classifier predicted a class correctly, divided by the total number of times it predicted this class. 
- Recall is the proportion of documents with a given class that were labelled correctly by the classifier. 
- The F1-score is the harmonic mean between precision and recall: $2*P*R/(P+R)$

The classification report below shows, for example, that the sports classes were quite easy to predict, while the computer and some of the politics classes proved much more difficult. 

### and Confusion matrix

# I've been using BERT model extensively for last few months and the performance of these models are really great!

# Text classification with BERT in PyTorch

2018 was an exciting year for Natural Language Processing. One of the most promising evolutions was the breakthrough of transfer learning. Models like Elmo Embeddings, ULMFit and BERT allow us to pre-train a neural network on a large collection of unlabelled texts. Thanks to an auxiliary task such as language modelling, these models are able to learn a lot about the syntax, semantics and morphology of a language. This knowledge can be put to good use: because they already know so much about language use, these models need much less labelled data to reach state-of-the-art performance on other tasks, such as text classification, sequence labelling or question answering. 

One of the most popular models is [BERT](https://arxiv.org/abs/1810.04805), developed by researchers at Google. BERT stands for Bidirectional Encoder Representations from Transformers. It uses the Transformer architecture to pretrain bidirectional "language models". By adding just one task-specific output layer, it is possible to use such a pre-trained BERT model on a variety of NLP tasks. In this notebook, we're going to investigate its performance on a sentiment analysis task, where the task is to predict whether a review is positive or negative. Unfortunately, we can't share the data, but you can easily plug in your own.

## BERT

Now we move to BERT. The team at [HuggingFace](https://github.com/huggingface) has developed a great Python library, [transformers](https://github.com/huggingface/transformers), with implementations of an impressive number of transfer-learning models in PyTorch and Tensorflow. It makes finetuning these models pretty easy. Let's first install this library. 

ple, we're going to investigate sentiment analysis on English. We'll therefore use the English BERT-base model.

### Preparing the data

Next we need to prepare our data for BERT. We'll present every document as a BertInputItem object, which contains all the information BERT needs: 

- a list of input ids. Take a look at the logging output to see what this means. Every text has been split up into subword units, which are shared between all the languages in the multilingual model. When a word appears frequently enough in a combined corpus of all languages, it is kept intact. If it is less frequent, it is split up into subword units that do occur frequently enough across all languages. This allows our model to process every text as a sequence of strings from a finite vocabulary of limited size. Note also the first `[CLS]` token. This token is added at the beginning of every document. The vector at the output of this token will be used by the BERT model for its sequence classification tasks: it serves as the input of the final, task-specific part of the neural network.
- the input mask: the input mask tells the model which parts of the input it should look at and which parts it should ignore. In our example, we have made sure that every text has a length of 100 tokens. This means that some texts will be cut off after 100 tokens, while others will have to be padded with extra tokens. In this latter case, these padding tokens will receive a mask value of 0, which means BERT should not take them into account for its classification task. 
- the segment_ids: some NLP task take several sequences as input. This is the case for question answering, natural language inference, etc. In this case, the segment ids tell BERT which sequence every token belongs to. In a text classification task like ours, however, there's only one segment, so all the input tokens receive segment id 0.
- the label id: the id of the label for this document.

### Evaluation method

Now it's time to write our evaluation method. This method takes as input a model and a data loader with the data we would like to evaluate on. For each batch, it computes the output of the model and the loss. We use this output to compute the obtained precision, recall and F-score. During training, we will print the simple numbers. When we evaluate on the test set, we will output a full classification report.

### Training

Now it's time to start training. We're going to use the AdamW optimizer with a base learning rate of 5e-5, and train for a maximum of 100 epochs. Here are some additional things to note: 

- Gradient Accumulation allows us to keep our batches small enough to fit into the memory of our GPU, while getting the advantages of using larger batch sizes. In practice, it means we sum the gradients of several batches, before we perform a step of gradient descent. 
- We use the WarmupLinearScheduler to vary our learning rate during the training process. First, we're going to start with a small learning rate, which increases linearly during the warmup stage. Afterwards it slowly decreases again.


