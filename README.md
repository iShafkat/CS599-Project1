# CS599-Project1
CS599 ( Machine Learning Research) Project 1

# Description:

In our project we are going to use dataset having information of email. This email is collected from office and personal email. Here as an input different feature of email will be used as input like percent of common words, length of capital latter etc. This is mainly a spam-based data set where output will be the class of spam or not. But the ultimate output of this project is a figure where cross-validation estimate of misclassification rate of tree size and cross validation. 

# Goal:

Goal of this project is to build Classification Decision Trees without using any Machine learning libraries (like Scikit Learn).
At the a figure will be generated having the results of misclassification rate vs tree size.

Build Multiclass decision Trees with Numerical features. We are using ID3 algorithm

IDE algorithm : https://en.wikipedia.org/wiki/ID3_algorithm.

# Data Set:

Spambase: is a binary classification task and the objective is to classify email messages as being spam or not. To this end the dataset uses fifty seven text based features to represent each email message. There are about 4600 instances. Here are what each feature means :

The last column of 'spambase.data' denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. Most of the attributes indicate whether a particular word or character was frequently occuring in the e-mail. The run-length attributes (55-57) measure the length of sequences of consecutive capital letters. For the statistical measures of each attribute, see the end of this file. Here are the definitions of the attributes:

48 continuous real [0,100] attributes of type word_freq_WORD = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

6 continuous real [0,100] attributes of type char_freq_CHAR] = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

1 continuous real [1,...] attribute of type capital_run_length_average = average length of uninterrupted sequences of capital letters

1 continuous integer [1,...] attribute of type capital_run_length_longest = length of longest uninterrupted sequence of capital letters

1 continuous integer [1,...] attribute of type capital_run_length_total = sum of length of uninterrupted sequences of capital letters = total number of capital letters in the e-mail

1 nominal {0,1} class attribute of type spam = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.
