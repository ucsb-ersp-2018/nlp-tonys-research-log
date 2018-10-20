# Tony's Research Log :octocat:

___

## Survey Paper: Gender Bias in Natural Language Processing
   ### Soft deadlines
   - [ ] 10/30: Outline of paper, finalize set of papers (we should probably try to finalize papers a week in advance though)
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

## Week 4 (10/18-10/24)
### Weekly Goals:
* Go to all meetings!
  - [ ] Attend both of the research methods lecture with Professor Mirza
  - [ ] Attend Professor William Wang's NLP research group meeting
  - [ ] Attend Intro to NLP lecture
  - [x] Attend lab meeting
* Learn about ML / NLP fundamentals
  - [ ] Learn about different types of ML classifiers

   ### Meeting minutes with Professor Wang and Mai
     #### [Meeting inputs](https://docs.google.com/document/d/1kTvtGg6FzSVyv7xTbk3uDzsbKDlY8iDJIGbARWSyIk0/edit?usp=sharing)
     #### [Meeting outputs](https://docs.google.com/document/d/1I66PzyQ8BvWalXakwMpZKOK9uR0sis5zxeFGHWztdH0/edit?usp=sharing)
     #### [Last Week's Meeting Minutes](https://docs.google.com/document/d/1D1hw3S-Pd6kZu8dMB0y19PN43qONlThHe9V1LNHE_lc/edit?usp=sharing)

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
  * **Performing Research:** It is an absolute privilege that I have the opportunity to contribute in a meaningful way to our world's understanding of computer science. Seeing what the brightest minds in the field are doing today and building upon that foundation of knowledge is just such a thrilling concept to me. I am incredibly eager to get started.
> What are you most nervous about in ERSP, and why?

I can see myself getting overwhelmed if I do not plan ahead accordingly. I recognize that the ERSP requires a lot of hard work, and it will take immense discipline and consistency to stay on top of everything.\

#### Log Reflection
> How did the logs differ in style (not just in content)?  What advantages do you see in one style over another?

I would say that both of the logs were quite informational and well-organized. Miranda's log was pretty impressive, and the design of her page was easier on the eyes. On the other hand, Adrian's log was more to the point, which can be beneficial if you're going over the log and short on time.
> How do you think the logs were useful, both to the researcher as well as those working with the researcher?

As a student, I can definitely see the importance of recording your work. The timeline serves as an excellent reference for the future, since it is impossible to remember every detail. It also communicates precisely where you are at in your research to anyone reading your log. I am sure that professors and researchers will find it helpful to know your progress so that they can give effective advice and adjust the project appropriately.
> Did the students keeping their logs seem to meet their goals?  Did they get better at meeting their goals over time?

Miranda had an explicit "Goals" and "Accomplished" section that I found quite helpful when reading her log. Adrian clearly defined his goals, but it was not always obvious whether or not he accomplished them. Both of them definitely showed progress and improvement over time. For example, Adrian appeared to gain much more out of his meetings with advisors and peers towards the end of the school year comppared to the start. Miranda was quite consistent throughout, but I noticed the difficulty of her goals gradually increased while she maintained a similar accomplishment rate.
