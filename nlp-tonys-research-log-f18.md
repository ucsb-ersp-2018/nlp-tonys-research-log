# Tony's Research Log :octocat:

___

## Survey Paper: Gender Bias in Natural Language Processing
   ### Soft deadlines
   - [x] 10/30: Outline of paper, finalize set of papers (we should probably try to finalize papers a week in advance though)
     - [Final set of papers](https://docs.google.com/document/d/1t4uOjNH62jzpmNbRx0N_OIGs_SQH2VLU4unq4ApMydQ/edit?usp=sharing)
   - [ ] 11/15: First draft of paper, 5-10 iterations before we submit
   ### Hard deadlines
   - [ ] 12/3: Abstract due
   - [ ] 12/10: Full Paper

___

## Compilation of Useful Resources
* [Artificial Intelligence Wiki](https://skymind.ai/wiki/)
* [Glossary of Terms](https://skymind.ai/wiki/glossary)
* [Dan Jurafsky NLP](https://www.youtube.com/watch?v=zfH2ADGtzJQ&index=2&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm)
  * [Personal Notes](https://docs.google.com/document/d/1gr1zTvuQSUCwtvOqNz9lR9EOSweczcqQiT3lIWphStQ/edit?usp=sharing)
* [Speech and Language Processing Course by Dan Jurafksy and James Martin](https://web.stanford.edu/~jurafsky/slp3/)
* [Stanford NLP Lecture Series](https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)


___


## Week 6 (11/1-11/7)
### Weekly Goals:
* Go to all meetings!
  - [ ] Attend both of the research methods lecture with Professor Mirza
  - [x] Attend Professor William Wang's NLP research group meeting and present Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings Paper
  - [x] Attend lab meeting with Mai
  - [x] Attend team meeting
* Learn about ML / NLP fundamentals
  - [x] Finish outline of literature review paper
  - [ ] Understand Mitigating Unwanted Bias with Adversarial Learning paper
  - [ ] Work on rough draft of survey paper with feedback from Mai and Professor Wang

### November 7, Wednesday (1.5 hours)
 - [x] Present PowerPoint on Debiasing Word Embeddings to Professor Wang and UCSB NLP group

[Feedback on Debiasing Word Embeddings](https://docs.google.com/document/d/1Fb-jKjkLvO8RhgQx80TkotEEvrIVefTpUYGhOYyTnDA/edit?usp=sharing)

### November 6, Tuesday (5 hours)
 - [x] Attend Intro to NLP Lecture
 - [x] Finish reviewing [Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
 - [x] Prepare for presentation tomorrow to Professor Wang and NLP Group

[Powerpoint Presentation on Debiasing Word Embeddings](https://drive.google.com/file/d/1_gWwPrUlpkcsh60ncbx_0sH9TDtCAKA5/view?usp=sharing)

#### Key Takeaways from Intro to NLP Lecture
 * Word Sense Disambiguation (WSD) primarily utilizes two kinds of feature vectors
   * Collocational features: features about words at specific positions near targer word, often limited to just word identity and POS
   * Bag-of-words: features about words that occur anywhere in the window regardless of position
   * Bootstrap WSD classifiers by building word sense classifiers with little training data
 * Semi-supervised learning
   * Yarowsky algorithm: Presents semi-supervised learning algorithm for WSD that can be applied to completely untagged text, uses decision lists
   * We need semi-supervised learning algorithms in NLP because linguistic data is extremely difficult to annotate and there is a lack of labeled training data
 * Properties of language
   * One sense per collocation: nearby words provide strong, consistent clues as to the sense of a target word
   * One sense per discourse: sense of a target word is highly consistent within a single document

### November 5, Monday (1.5 hours)
 - [x] Review [Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
 - [x] Coordinated with Professor Mirza and Mai about our goals and assignments over the quarter

### November 4, Sunday (1.5 hours)
 - [x] Work on survey paper
 - [x] Review [Semantics derived automatically from language corpora contain human-like biases](http://science.sciencemag.org/content/356/6334/183)

### November 3, Saturday (4 hours)
 - [x] Work on survey paper
 - [x] Read [Getting Gender Right in Neural Machine Translation](http://aclweb.org/anthology/D18-1334)

### November 2, Friday (1.5 hours)
 - [x] Work on Pros and Cons and Future Direction of survey paper
 
[Pros / Cons / Future Direction](https://docs.google.com/document/d/1URjDeIiGO__IfORDHKG_m_SO3Pl8t8pOOHmtGwjJ_Ls/edit?usp=sharing)

### November 1, Thursday (1.5 hours)
 - [x] Meeting with team and Mai -- William is at the EMNLP conference
 - [x] Contribute to survey paper

## Week 5 (10/25-10/31)
### Weekly Goals:
* Go to all meetings!
  - [x] Attend both of the research methods lecture with Professor Mirza
  - [ ] Attend ~~Professor William Wang's NLP research group meeting~~ [meeting cancelled due to Professor Wang attending EMNLP conference]
  - [x] Attend lab meeting with Mai and Professor Wang
  - [x] Attend team meeting
* Learn about ML / NLP fundamentals
  - [ ] Finish outline of literature review paper
  - [x] Finalize list of papers to use

### Meeting minutes with Professor Wang and Mai
  * ####    [Meeting Inputs](https://docs.google.com/document/d/14KhueElt5UMloa3rwZ0bKzZ4ZgDiGWSFJybp5XDRujc/edit?usp=sharing)
  * ####    [Meeting Outputs](https://docs.google.com/document/d/1QsWpE2g5CV5EGhrQdAF-bm-ULHqKIJE9qiAOl_C5_g4/edit?usp=sharing)
  * ####    [Last Week's Meeting Minutes](https://docs.google.com/document/d/1I66PzyQ8BvWalXakwMpZKOK9uR0sis5zxeFGHWztdH0/edit?usp=sharing)

### October 31, Wednesday (1.5 hours)
 - [x] Continue reading [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/pdf/1801.07593.pdf)
 - [x] Read more about adversarial networks
 
#### [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/pdf/1801.07593.pdf)
 * Information is quite dense, difficult to sift through
 * Concept of GANs make sense, understanding how they apply to NLP is gradually becoming more clear but it will take time and effort

### October 30, Tuesday (1 hour)
 - [x] Read Should Computer Scientists Experiment More?
 - [x] Start reading about [Deep Reinforcement Learning](https://skymind.ai/wiki/deep-reinforcement-learning)

### October 29, Monday (1.5 hours)
 - [x] Finish Literature Search Part 2
 - [x] Continue reading [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/pdf/1801.07593.pdf)
 
 [Literature Search Part 2](https://docs.google.com/document/d/1Csxf0NTG1gNC9FSfE4ExCjFMECZMJK6qdytw9Gn2Oy8/edit?usp=sharing)

### October 28, Sunday (2 hours)
 - [x] Formalize Literature Search Part 1
 - [x] Work on Literature Search Part 2
 
 [Literature Search Part 1](https://docs.google.com/document/d/1JH_mQShiAsuMc_zww5af2VDbF_fF5XMGjvMRH9ouwTw/edit?usp=sharing)
 
 [Literature Search Part 2](https://docs.google.com/document/d/1JH_mQShiAsuMc_zww5af2VDbF_fF5XMGjvMRH9ouwTw/edit?usp=sharing)

### October 27, Saturday (3 hours)
 - [x] Read [Reducing Gender Bias in Abusive Language Detection](https://arxiv.org/abs/1808.07231)
 - [x] Skim [Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/pdf/1801.07593.pdf) and [Avoiding Discrimination through Causal Reasoning](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)
 - [x] Attend team meeting
 
[Tentative Survey Paper Outline](https://docs.google.com/document/d/1wATwfczLcQ1kCHdylPhD_Il56Zfo5CeHLIMBMD4gsTw/edit?usp=sharing)
[Current Progress on Outline](https://docs.google.com/document/d/1sLckNjZThHiaYXWIy8-rjziD220nUSMqa196p6xxSyE/edit?usp=sharing)

### October 26, Friday (1 hour)
 - [x] Meeting with Mai, took notes on paper

### October 25, Thursday (3 hours)
 - [x] Read [Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems](http://aclweb.org/anthology/S18-2005.pdf)
 - [x] Read [Semantics derived automatically from language corpora necessarily contain human biases](http://www.cs.bath.ac.uk/~jjb/ftp/CaliskanSemantics-Arxiv.pdf)
 - [x] Skim [Rejecting the gender binary: a vector-space operation](http://bookworm.benschmidt.org/posts/2015-10-30-rejecting-the-gender-binary.html)
 - [x] Attend meeting with Professor Wang and Mai
 
#### Sentiment Analysis, Semantics, and Rejecting the Gender Binary Notes
 * Took notes [here](https://docs.google.com/document/d/1qlLdOs6ku93qZ2fENe3YL5Ikp0yF02XpZ8ExjYzCPHg/edit) and [here](https://docs.google.com/document/d/1uC8kRC9huDNK4Aem18zAQpNfJGKetWrDbN5p8eqYCDA/edit)
 
#### Meeting with Mai and Professor Wang
 * [Meeting inputs](https://docs.google.com/document/d/14KhueElt5UMloa3rwZ0bKzZ4ZgDiGWSFJybp5XDRujc/edit?usp=sharing)
 * [Meeting outputs](https://docs.google.com/document/d/1QsWpE2g5CV5EGhrQdAF-bm-ULHqKIJE9qiAOl_C5_g4/edit?usp=sharing)

## Week 4 (10/18-10/24)
### Weekly Goals:
* Go to all meetings!
  - [x] Attend both of the research methods lecture with Professor Mirza
  - [x] Attend Professor William Wang's NLP research group meeting
  - [x] Attend lab meeting with Mai and Professor Wang
  - [x] Attend team meeting
* Learn about ML / NLP fundamentals
  - [x] Finish chapter 4 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?)
  - [x] Create a simple linear regression model using ML on Python
  - [x] Learn about different types of ML classifiers

### Meeting minutes with Professor Wang and Mai
  * ####    [Meeting Inputs](https://docs.google.com/document/d/1kTvtGg6FzSVyv7xTbk3uDzsbKDlY8iDJIGbARWSyIk0/edit?usp=sharing)
  * ####    [Meeting Outputs](https://docs.google.com/document/d/1I66PzyQ8BvWalXakwMpZKOK9uR0sis5zxeFGHWztdH0/edit?usp=sharing)
  * ####    [Last Week's Meeting Minutes](https://docs.google.com/document/d/1D1hw3S-Pd6kZu8dMB0y19PN43qONlThHe9V1LNHE_lc/edit?usp=sharing)

### October 24, Wednesday (1.5 hours)
 - [x] Attend Professor Wang's NLP research group meeting
 - [x] Attend research methods lecture
 - [x] Find target articles for literature review paper

### October 23, Tuesday (2.5 hours)
 - [x] Learn about Naive Bayes Classifier
 - [x] Watch video on [Bias vs Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
 - [x] Watch chapter 6 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?)

#### Terminology
 * Accuracy: total number of correct predictions / total number of predictions
 * Precision: how many selected items are relevant? true positives / (true positives + false positives)
 * Recall: how many relevant items are selected? true positives / (true positives + false negatives)
   * F measure: combined measure that assesses the precision/recall tradeoff: 1/(alpha*(1/P) + (1 - alpha)\*(1/R))
   * Usually we use F1 measure (alpha is 1/2), so precision and recall are weighted equally
 * Bias: "error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting)"
 * Variance: "error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting)"
   * Ideally we have a low-bias, low-variance model, but is difficult due to bias-variance tradeoff
 * Naive Bayes: conditional probability model that simplifies assumptions and uses Bayes' theorem to find output with maximum probability 
 * Confusion matrix: a table with actual and predicted outcomes of a model, makes it easy to identify mistakes

### October 22, Monday (1 hour)
 - [x] Create Google Drive to organize documents
 - [x] Work on [Group Reflection](https://docs.google.com/document/d/1H_V2T2gNRewMRvP6ARWIC2B2QxkSvJ1BujZ9456PTow/edit?usp=sharing)
 - [x] Skim abstracts of papers on papers on bias in NLP
 - [x] Attend research methods lecture

### October 21, Sunday (1.5 hours)
 - [x] Identify Research Problems

#### Identifying Research Problems
> Bs well as more specific technical problemsrainstorm the various problems that your broader research group is focused on. Try to include both high-level problems as well as more specific technical problems.

I am not quite sure what my "broader research group" refers to specifically, but I will assume it refers to the broad group of people researching bias in natural language processing (NLP). To my understanding, bias in NLP is a relatively nascent concept. It refers to how machines trained on text corpora may unknowingly capture human biases and stereotypes. Consequently, other machines or algorithms that rely upon these NLP models in downstream applications may transmit these biases and stereotypes to the user, which could easily propagate bigotry and prejudice. Researchers who are attempting to remove bias in NLP have utilized a variety of approaches, ranging from manipulating word embeddings, data augmentation, restricting model output, and anonymizing named entities. However, there has not yet been one single best approach because there are many stages at which text is processed and analyzed, meaning that what works at one stage may not be so effective at another. Furthermore, it is difficult to identify to what extent NLP models reflect or amplify bias present in society.

One of the papers our team has read discussed the idea of removing the gender subspace of gender-neutral words from the overall vectorspace of words. While certainly a promising solution, the consequence of methods along these lines risks destroying some semantic meaning of these gender-neutral words. For example, *clock* is a gender-neutral word, but it has associations with the word *grandfather*. By removing clock's gender subspace, then an NLP model would predict *grandmother* and *grandfather* with equal probability to precede *clock*, which does not reflect the actual English language. Other people have tried anonymizing named entities and augmenting the corpora by swapping gendered words, but once again, it is difficult to test the degree to which these approaches have affected semantic meaning and removed "bias" (although there have been methods to quantify bias).

Since our team's role is to assess the performance of proposed methods to remove bias in NLP, we are deeply involved with both the high-level problems as well as the more specific technical ones. I would say, though, that we are more concerned with the big picture and how current approaches compare to one another and what the future directions of the field might be.

#### Reflection
> In your log, write a short reflection on your experience in ERSP so far. What do you like most about the work you have been doing?  What do you like least?  What is the biggest concern or question you have?

I love, love, love the satisfaction I feel from contributing meaningfully to the advancement of a particular field of computer science. So far, it has been such an incredible experience to be able to pick the minds the outstanding faculty at the university and collaborate with my extremely bright peers. To be frank, coming into the program, I wasn't sure if I was going to enjoy learning about gender bias in natural language processing simply because it seemed like such an abstract problem. However, understanding how others have tranformed this topic into concrete numbers and statistics has appealed to me greatly. I find it highly fascinating to see how we can represent nebulous ideas as mathematical concepts. Furthermore, I recognize the significance and consequences of bias in natural language processing, such as how it can propagate stereotypes if left unchecked.

It has certainly been a lot of hard work, time, and effort, but I truly believe it will pay big dividends in the future. Our team's goal by the end of this year is to publish a literature review paper to a top conference. Although task is not easy, I know that I can be proud of my work regardless of the outcome. I can't wait to see what the future holds.

### October 20, Saturday (3 hours)
 - [x] Read [A Survey on Information Retrieval, Text Categorization, and Web Crawling](https://arxiv.org/ftp/arxiv/papers/1212/1212.2065.pdf)
 - [x] Meeting with team members discussing papers and direction of literature review paper

### October 19, Friday (3 hours)
 - [x] Watch 4.6 - 4.8 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?)
 - [x] Read [Social Bias in Elicited Natural Language Inferences](http://www.aclweb.org/anthology/W17-1609)
 - [x] Learn about [Pointwise Mutual Information](https://www.youtube.com/watch?v=swDoFpuHpzQ) vs [Mutual Information](https://www.youtube.com/watch?v=U9h1xkNELvY)

### October 18, Thursday (4 hours)
 - [x] Meeting with Mai and Professor Wang
 - [x] Meeting with teammates
 - [x] Read about [feature scaling](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)
 - [x] Create a simple linear regression [experience vs salary model](https://github.com/tonysun9/machine-learning-models/blob/master/simpleLinearRegression.py) on Python

#### Meeting with Mai and Professor Wang
 * [Meeting inputs](https://docs.google.com/document/d/1kTvtGg6FzSVyv7xTbk3uDzsbKDlY8iDJIGbARWSyIk0/edit?usp=sharing)
 * [Meeting outputs](https://docs.google.com/document/d/1I66PzyQ8BvWalXakwMpZKOK9uR0sis5zxeFGHWztdH0/edit?usp=sharing)

#### Importance of Feature Scaling
 * Scaling ensures that just because some features are big it won't lead to the machine learning model using them as a main predictor
 * Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one
   * Standardization is an important preprocessing step for ML models such as SVM, K-nearest neighbors, and logistic regression

## Week 3 (10/11-10/17)
### Weekly Goals:
* Go to all meetings!
  - ~~[ ] Attend Yu Xiang's SMLRG seminar~~ (schedule conflict)
  - [x] Attend both of the research methods lecture with Professor Mirza
  - [x] Attend Professor William Wang's NLP research group meeting
  - [x] Attend both Intro to NLP lectures
  - [x] Attend lab meeting
* Learn about ML / NLP fundamentals
  - [x] Read and discuss two new papers on gender bias in NLP
  - [x] Explore Python and create a simple ML model

#### [This Week's Meeting Minutes](https://docs.google.com/document/d/1D1hw3S-Pd6kZu8dMB0y19PN43qONlThHe9V1LNHE_lc/edit?usp=sharing)

### October 17, Wednesday (1.5 hours)
 - [x] Attend NLP research group meeting
 - [x] Read articles on [logistic regression](https://skymind.ai/wiki/logistic-regression) and [backpropagation](https://skymind.ai/wiki/backpropagation) and understand how these topics relate to gradient descent
 - [x] Attend research methods lecture
 
#### NLP Research Group Meeting
 * Dual learning machine translation
   * I didn't fully understand the paper that was presented, but it was pretty interesting to listen in to the discussion of people who are at the forefront of the field

### October 16, Tuesday (2.5 hours)
 - [x] Attended Intro to NLP lecture
 - [x] Learned about gradient descent
 - [x] Watch videos on particle swarm optimization [here](https://www.youtube.com/watch?v=JhgDMAm-imI) and [here](https://www.youtube.com/watch?v=ckbdXbbNNNA)
 
#### Key Takeaways from NLP lecture
 * POS tagging approaches
   * Rule-based: human crafted rules based on lexical and other linguistic knowledge
   * Learning-based: trained on human annotated corpora like the Penn Treebank, e.g. hidden markov model, conditional random field
 * Probabilistic sequence models allow integrating uncertainty over multiple, interdependent classifications and collectively determine the most likely global assignement
 
[Gradient Descent Notes](https://docs.google.com/document/d/1z5suHbjPTEZp_ZZ9mJPe1R67Re6RvDhRsRw66cdL55Y/edit?usp=sharing)

### October 15, Monday (2 hours)
 - [x] Read [Gender Bias in Coreference Resolution (Zhao et al. 2018)](http://web.cs.ucla.edu/~kwchang/publications/ZWYOC18.html)
 - [x] Discuss paper with team to gain a deeper understanding of methodology
 - [x] Read about [gradient descent](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0) for Wednesday presentation

#### Key takeaways from [Gradient Descent Article](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)
 * Gradient descent is a minimization algorithm that minimizes a given cost function
   * A gradient measures the change in all weights in response to some change in error/cost, represents the slope (partial derivative) of a function
   * Find local minimum of function J(w,b) by adjusting parameters w and b
 * The next position is your current position minus some factor times the gradient term
   * Take larger steps when slope is steeper, smaller steps when slope is flatter
 * Mini batch gradient descent combines the utility of stochastic gradient descent and efficiency of batch gradient descent
![Gradient Descent](https://cdn-images-1.medium.com/max/703/1*t4aYsxpCqz2eymJ4zkUS9Q.png)

### October 14, Sunday (5 hours)
 - [x] Complete second pass-through of [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520)]
 - [x] Explore Python and some machine learning libraries
 - [x] Create a [restaurant review machine learning model](https://github.com/tonysun9/machine-learning-models/blob/master/restaurantReviews.py "Restaurant Review ML Model") using naive bayes classifier
 
 [Reading log](https://docs.google.com/document/d/1e-VLsE-arXcU3DEQ2v1Zjmsf9z9Kggu6DjFRJdLwRQ8/edit?usp=sharing) for [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520)

#### [Restaurant Review ML Model](https://github.com/tonysun9/machine-learning-models/blob/master/restaurantReviews.py "Restaurant Review ML Model")
 * Set up Python IDE using Spyder
 * Used a naive bayes classifier to classify whether a review is positive (1) or negative (0)
 * Used continuous bag of words (CBOW) model to represent restaurant reviews
 * Addressing sparse matrix (matrix will have many 0's due to the ratio between unique words per review and size of vocabulary of corpus)
   * Normalize reviews through regular expressions, lower-case letters, and stopwords
   * Expand each review into its components to enable stemming using PorterStemmer
   * Collapse back into a string and append into new and simplified corpus
 * Use only the 1500 most common words to create the CBOW model so that words that appear infrequently (e.g. Steve) will not affect the way the naive bayes classifier learns
 * Training data of 800 reviews, test data of 200 reviews, success rate of 73% -- accuracy will improve with size of training data

### October 13, Saturday (2 hours)
 - [x] Meet with team to discuss [Men Also Like Shopping](https://arxiv.org/pdf/1707.09457.pdf) and [Learning Gender-Neutral Word Embeddings](https://arxiv.org/pdf/1809.01496.pdf)

### October 12, Friday (4.5 hours)
 - [x] Read [example of survey paper](http://aclweb.org/anthology/D15-1021)
 - [x] Read and annotate [Men Also Like Shopping](https://arxiv.org/pdf/1707.09457.pdf)
 - [x] Read [Joint Inference for NLP](http://www.aclweb.org/anthology/W09-1101)
 - [x] Read [Learning Gender-Neutral Word Embeddings](https://arxiv.org/pdf/1809.01496.pdf)

### October 11, Thursday (3 hours)
 - [x] Attend Intro to NLP Lecture
 - [x] Review [word embeddings](https://skymind.ai/wiki/word2vec)
 - [x] Meeting with team members
 - [x] Meeting with Professor Wang and Mai
 - [x] Call Yuxin to discuss team development
 
#### Intro to NLP Lecture
 * Talked about Naive Bayes Classifier and Voted Perceptrons
 * Played perceptron game to understand perceptrons better
   * Perceptrons don’t need to retrain entire model if given new data; however, they can’t mesaure uncertainty (probability), only outputs positive or negative
   * Voted perceptrons are an ensemble of perceptrons
 * Review of gradient descent
   * Pros: simple and often quite effective on ML tasks, is scalable to larger tasks
   * Cons: only applies to smooth functions (differentiable), and method might find a local minimum rather than global one, no guarantee for optimal solution
 * Terms to work on: perceptron, voted perceptron, naive bayes classifier, gradient ascent for linear classifiers, stochastic gradient descent
 
 [Meeting Notes](https://docs.google.com/document/d/1D1hw3S-Pd6kZu8dMB0y19PN43qONlThHe9V1LNHE_lc/edit?usp=sharing)

## Week 2 (10/4-10/10)
### Weekly Goals:
* Go to all meetings!
  - ~~[ ] Attend Yu Xiang's SMLRG seminar~~ (seminar cancelled)
  - [x] Attend both of the research methods lecture with Professor Mirza
  - [x] Attend Professor William Wang's NLP research group meeting
  - [x] Attend lab meeting
* Learn about ML / NLP fundamentals
  - [x] Watch [3Blue1Brown's video series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  - [x] Watch chapter 1, 2, and 4-1 to 4-5 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?

### October 10, Wednesday (1.5 hours)
 - [x] Attended Professor William Wang's NLP research group meeting
 - [x] Transcribe annotations from [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520)
 
#### Research Group Meeting
 * PhD student Yu Xiang talked about state-of-the-art methodologies in Reinforcement Learning, followed up by a few graduate students who gave a PowerPoint on their current research
 * Didn't understand most of the information, but did my best to follow along
   * I recognize that the learning process will just take time and effort, not worried about my lack of knowledge yet
 * Terms to learn: Reinforcement learning, Kalman filter / LQR, Contextual Bandits, Multi-arm bandits / Bandits

#### Key Takeaways from [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520)
 * [Reading log](https://docs.google.com/document/d/1e-VLsE-arXcU3DEQ2v1Zjmsf9z9Kggu6DjFRJdLwRQ8/edit?usp=sharing)
 * The goal of the paper is to debias word embeddings by removing gender stereotypes while keeping the useful aspects intact
 * Gender stereotype is empirically demonstrated through both word embeddings and crowd-sourced human evaluation
   * Geometric understanding of bias is captured by both cosine similarity and delta threshold
     * Delta threshold ensures the generated analogy have semantic similarity, and the cosine measures the extent to which gender determines the relationship between the two words
   * Human evaluation is determined by crowd-sourced surveys
 * Turns out that word embeddings do a good job of capturing gender stereotypes as viewed by humans
 * Have yet to fully understand methodology and immediate consequence of the paper

### October 9th, Tuesday (3 hours)
 - [x] Attended Professor Wang's Intro to NLP lecture
 - [x] Continue reading [Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)
 
#### Intro to NLP Lecture
 * Talked about Naive Bayes and Voted Perceptron and used smoothing terminologies
 * Reducing a very large d-dimensional word vector that represents a sentence
   * We can get rid of all the zeros in the word vector (words in the vector that are not found in the sentence) by simply recording the the index of the words and their count 
 * Recognized a few concepts, but still a long, long way from full comprehension

### October 8th, Monday (2.5 hours)
 - [x] Watch 4.3 - 4.5 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?)
 
#### Key Takeaways from 4.3 (Evaluation and Perplexity), 4.4 (Generalization and Zeros), and 4.5 (Add-one Laplace smoothing)
 * Good language models assign higher probability to “real” or “frequently observed sentences” rather than “ungrammatical” or “rarely observed” sentences
 * Extrinsic evaluation of N-gram models measures performance based on external task
   * However, extrinsive (in-vivo) evaluation consumes a lot of resources and time
 * Intrinsic evaluation measures something that’s intrinsic about language models and not about any particular application
   * Perplexity is a common metric of intrinsic evaluation, measures how well the model can predict the next word given some context
   * Better models have lower perplexity and improve with length of n-gram
   * However, models can't evaluate perplexity for 0 probability n-gram because if the string of words never occurs in the training data, the model will not predict a previously unseen string when evaluating test data
 * A simple way to address this issue is to siphon some of the probability present in other n-grams (let's say bigram for our example) and give it to the bigrams with 0 probability
   * Add-one (Laplace) smoothing adds 1 to each possible bigram; however, this is not a scalable solution if the number of 0 probability bigrams is large relative to the possible set of bigrams
 * Have questions about bigram --> unigram and + V in Laplace smoothing
 

### October 7th, Sunday (3.5 hours)
 - [x] Read part of [Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)
 - [x] Read article on [AI vs. ML vs. DL](https://skymind.ai/wiki/ai-vs-machine-learning-vs-deep-learning)
 - [x] Read about Random Forest [here](https://skymind.ai/wiki/random-forest) and [here](https://en.wikipedia.org/wiki/Random_forest)
 - [x] Read [A Beginner's Quide to Word2Vec and Neural Word Embeddings](https://skymind.ai/wiki/word2vec)
 - [x] Set up meeting time with group
 
#### Key Takeaways from [AI vs. ML vs. DL](https://skymind.ai/wiki/ai-vs-machine-learning-vs-deep-learning)
 * DL is a subset of ML which is a subset of AI
 * Artificial Intelligence: programming computers to perform intelligent tasks
   * Can be as simple as a series of if-else statements, otherwise known as expert systems or knowledge graphs
   * John McCarthy is the father of AI
 * Machine Learning: dynamic programs that can modify themselves without human intervention when exposed to more data
   * Might be done by minimizing a loss / error function or maximizing an objective function, is basically an optimization function
   * Pioneer Arthur Samuel used ML to create a checkers program that could play better than he could
 * Deep Learning: can refer either deep artificial neural networks or deep reinforcement learning
   * Deep neural networks contain more than one hidden layer, which allows it to form more complex features
   * Require better hardware (GPU), more training data and time, and more overall computational intensity, but results in higher accuracy
   * DeepMind's AlphaGo algorithm is a well-known application of deep learning
   
#### Brief Overview of Random Forest (still don't fully understand)
 * Random forests are composed of many decision trees
 * However, they correct for overfitting, a common flaw of decision trees
   * Average multiple deep decision trees, which reduces variance but increases bias

#### Key Takeaways from [Word2Vec](https://skymind.ai/wiki/word2vec)
 * Two-layer neural network that takes in text corpus and outputs a vocabulary comprised of a set of vectors (labels / features) that comprise a vector space
   * Can be applied to any discrete object, such as genes, likes, and playlists
 * Neural Word Embeddings are vectors that are made up of numbers that correspond to the features of words, such as context, gender, or distance
   * Relationships between vectors give clue to the meaning of the words, e.g. Rome - Italy = Beijing - China
   * Similar words are grouped in a cluster, measured specifically by cosine similarity (v⋅w=∥v∥∥w∥cosθ)
 * Can be trained using either a continuous bag of words (CBOW) or skip-gram
   * CBOW: using context to predict a word
   * Skip-gram: using a word to predict the context
     * A n-gram but with dropped items, has shown to be more accurate than the CBOW model
> When the feature vector assigned to a word cannot be used to accurately predict that word’s context, the components of the vector are adjusted. Each word’s context in the corpus is the teacher sending error signals back to adjust the feature vector. The vectors of words judged similar by their context are nudged closer together by adjusting the numbers in the vector.

Word Embeddings Visualized, Word2Vec (skip-gram model)
![Word Embeddings](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/06062705/Word-Vectors.png)

### October 6th, Saturday (1.5 hours)
 - [x] Review yesterday's [notes on NLP](https://docs.google.com/document/d/1gr1zTvuQSUCwtvOqNz9lR9EOSweczcqQiT3lIWphStQ/edit?usp=sharing)
 - [x] Watch 4.1 - 4.2 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?v=3Dt_yh1mf_U&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=1)
 - [x] Finished [3B1B's series on Neural Networks]((https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) -- watched backpropagation calculus, had watched the other videos previously

#### Key Takeaways from 4.1 (Introduction to N-grams), 4.2 (Estimating N-gram Probabilities)
 * Probabalistic language models assign a probability to a sentence
   * Spell correction, e.g. fifteen minutes from > fifteen minuets from
   * Speech recognition, e.g. I saw a van > eyes awe of an
 * Markov Assumption: simplifies the conditional probability chain rule model applied to sentences to increase usable data
 * Unigram, bigram, and n-gram models
   * Prefix represents length of Markov model
   * Tradeoff between size of n and amount of usable data for language models

#### Basic Understanding of [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)
 * Use cost function to minimize error
 * Derivative of the change function provides the slope for the gradient descent

### October 5th, Friday (3.5 hours)
 - [x] Watch 1.1 - 2.5 of [Dan Jurafsky's Series on NLP](https://www.youtube.com/watch?v=3Dt_yh1mf_U&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=1)
 
Full notes [here](https://docs.google.com/document/d/1gr1zTvuQSUCwtvOqNz9lR9EOSweczcqQiT3lIWphStQ/edit?usp=sharing), formatting idea from [Cynthia's reading notes](https://docs.google.com/document/d/1joi0W6lgbNccDmH9SjlzKFVs1GzOCykV9FxEBf8ysfs/edit)

#### Key Takeaways from 1.1 (Course Intro), 2.1 (Regular Expressions), and 2.2 (Regular Expressions in Practical NLP)
 * NLP is useful for solving problems like task answering, information extraction, sentiment analysis, and machine translation
 * Ambiguity makes NLP hard ("Crash Blossoms")
   * E.g. 1: Red Tape Holds Up New Bridges is an actual title of a news article, but "Holds Up" refers to a delay rather than physically holding up the bridge
   * E.g. 2: Fed raises interest rates 0.5%, humans clearly understand that the interest rate is rising, but a parser might interepret "raises" or even "rates" as the verb
 * Things that make NLP hard: non-standard English, segmentation issues, idioms, neologisms, world knowledge, and tricky entity names
 * Regular expressions are a formal language for specifying text strings
   * Use disjunctions to narrow down your search
   * E.g. ((\+\+?)?[0-9]{2-4}\.)?[0-9]{2,4}\.[0-9]{3,4}\.[0-9]{3,5} is a regular expression that captures phone numbers
     * ((\+\+?)?[0-9]{2-4}\.)? represents an optional area code
     * [0-9]{2,4}\.[0-9]{3,4}\.[0-9]{3,5} represents the standard phone number separated by some character

#### Key Takeaways from 2.3 (Word Tokenization), 2.4 (Word Normalization and Stemming), and 2.5 (Sentence Segmentation and Decision Trees)
 * NLP tasks involve text normalization, which is comprised of segmenting/tokenizing words in running text, normalizing word formats, and segmenting sentences in running text
 * Type: an element of the vocabulary; Token: an instance of that type in running text
   * Church and Gale (1990) suggested that |V| > O(N1/2), where N denotes number of tokens and V denotes vocabulary / set of types
 * Word segmentation can be done through an algorithm called Maximum Matching
   * This method does not work well for English but actually performs pretty well for Chinese because words in Chinese have pretty consistent character length
 * Methods of word normalization include case folding, lemmatization, and stemming, which have applications in information retrieval
 * Decision trees (basically a flowchart / a series of if-else statements)have applications in sentence segmentation
   * Features in a decision tree can be applied to other classifiers like SVM, neural nets, logistic regression

 #### Things to Work On:
 - What is a standard convolutional network, and how does it differ from an inverse convolutional network?
 - Get a deeper understanding of an autoencoder and a variational autoencoder
 - Understand ML code
 - Using terminal commands to parse through text files
 - Review notes
 - Understand SVM, neural nets, logistic regression

### October 4th, Thursday (0.5 hours)
 - [x] Read [How to Read an Engineering Research Paper](http://cseweb.ucsd.edu/~wgg/CSE210/howtoread.html)
 - [x] Attended group meeting with Mai and team

#### Key Takeaways from "How to Read an Engineering Research Paper"
 * The motivation for a research paper is two-fold: it solves a problem that exists in the world, and it approaches / solves the problem in a way that has not been done yet
 * The body of the paper will usually provide the details of the proposed solution and the author's analysis of the given idea, which includes contributions, future directions, and unanswered questions
 * Griswold provides a [cheat sheet](http://cseweb.ucsd.edu/~wgg/CSE210/paperform.pdf) of questions to answer as you read through a given research paper

___

## Week 1 (09/27-10/3)
### Weekly Goals:
* Go to all meetings!
  - [x] Attend Yu Xiang's SMLRG seminar
  - [x] Attend both of the research methods lecture with Professor Mirza
  - [x] Attend Professor William Wang's NLP research group meeting
* Create and update research log
  - [x] Set up log and read previous examples
  - [x] Learn about [Github Markdown](https://guides.github.com/features/mastering-markdown/)
* Learn about ML / NLP fundamentals
  - [x] Read [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://skymind.ai/wiki/generative-adversarial-network-gan)
  - [ ] Watch [3Blue1Brown's video series on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### October 3rd, Wednesday (1.5 hours)
- [x] Attended William Wang's NLP research group meeting
- [x] Read [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://skymind.ai/wiki/generative-adversarial-network-gan)
- [x] Learned how to [upload local images for GFM](https://www.youtube.com/watch?v=nvPOUdz5PL4)
- [x] Created and set up Slack channel for team communication

#### Key Takeaways from "A Beginner's Guide to GANs"
 * Relationship b/w generator and discriminator is like actor-critic / forger-cop
 * Given a set of features, the discriminator predicts the label, aka p(y|x) where y is the label and x are the features
 * Vice versa for generator: given a label or D-dimensional noise vector, it attempts to predict the features and attempts to trick the discriminator into thinking that the output of the generator is from the real training set
 * This escalates until the generator produces items that are indistinguishable from the real training set, and the discriminator outputs 0.5 probability of being real for each item
 
![gan_framework](https://user-images.githubusercontent.com/36688734/46435398-1bd03e00-c70b-11e8-9380-ecad610ef2e4.png)

#### Things to Work On:
 - What is a standard convolutional network, and how does it differ from an inverse convolutional network?
 - Get a deeper understanding of an autoencoder and a variational autoencoder
 - Understand ML code

### October 2nd, Tuesday (2.5 hours)
- [x] Set up log
- [x] ERSP Initial Thoughts
- [x] Log Reflection
- [x] Read Miranda's and Adrian's log
- [x] Learn about [Github Markdown](https://github.github.com/gfm/) (seems like HTML for Github)

#### ERSP Initial Thoughts
> What are you most excited about in ERSP, and why?

Ahhh!! There are so, so many things that I'm excited about that it's hard to pinpoint just one thing, but I'll try to summarize the main points of interest in a few bullet points.
  * **Learning about Machine Learning and NLP:** When Deep Blue first beat Kasparov over twenty years ago, the program pretty much relied on brute force. So when Google's AlphaZero came out and surpassed the level of human players, to say I was fascinated would be an understatement. To me, these new programs are merely a glimpse of the potential and future of machine learning, which is why I cannot wait to learn more about it.
  * **Working with Esteemed Faculty and Talented Peers:** I am certainly super excited to perform hands-on, in-depth research with outstanding faculty in Professor Mirza and Professor William Wang as well as highly talented colleagues. I know that working alongside these amazing people will inspire me to constantly improve. I will definitely be taking advantage to pick their brains for knowledge, ideas, and advice.
