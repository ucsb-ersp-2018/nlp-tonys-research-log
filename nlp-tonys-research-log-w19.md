# Tony's Research Log :octocat:

___

[Google Drive](https://drive.google.com/drive/u/0/folders/17hVFvizRJ3pGERHZFLJzvudsoHoiFjSq) with the team's notes and to-do's

___

## Compilation of Useful Resources
* [Artificial Intelligence Wiki](https://skymind.ai/wiki/)
* [Glossary of Terms](https://skymind.ai/wiki/glossary)
* [Dan Jurafsky NLP](https://www.youtube.com/watch?v=zfH2ADGtzJQ&index=2&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm)
  * [Personal Notes](https://docs.google.com/document/d/1gr1zTvuQSUCwtvOqNz9lR9EOSweczcqQiT3lIWphStQ/edit?usp=sharing)
* [Speech and Language Processing Course by Dan Jurafksy and James Martin](https://web.stanford.edu/~jurafsky/slp3/)
* [Stanford NLP Lecture Series](https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)

___

## Week 5 (2/4-2/10)

### February 6, Tuesday (0.5 hours)
- [x] Look through a bit of the [BERT github](https://github.com/google-research/bert)

### February 5, Monday (1.5 hours)
- [x] Attend deep learning reading group meeting
- [x] Review chapters 1 and 2 of the deep learning book by Goodfellow

___

## Week 4 (1/28-2/3)

### February 3, Sunday (3 hours)
- [x] Update Github log
- [x] Go over last quarter's research paper to identify methods of debiasing that we can apply to BERT (since BERT produces word embeddings) -- removing gender component for multiple classes seems like an interesting direction

### February 2, Saturday (1.5 hours)
- [x] Briefly go over chapters 1 and 2 of [Deep Learning Book](https://www.deeplearningbook.org/)

* I liked example of separating cartesian vs polar coordinates as inspiration for different ways of approaching the same problem

### February 1, Friday (1 hour)
- [x] Meet with team and May for ERSP meeting

![Week 3 ERSP_Tony](https://user-images.githubusercontent.com/36688734/52200699-59aab380-281e-11e9-9522-26417eecc6f4.png)

* Went to last quarter's log to review how to upload an image!

### January 30, Wednesday (3.5 hours)
- [x] Read [BERT paper](https://arxiv.org/pdf/1810.04805.pdf) and [BERT article](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

Key Takeaways about BERT:
 * BERT is used in transfer learning -- namely, it produces word embeddings that are used as features in other models (as opposed to traditionally just producing feature embeddings)
 * BERT uses a Transformer, which is an "attention mechanism that learns contextual relations between words in text"
   * Does so via an encoder that reads the text input and a decoder that produces a prediction for the task
 * Traditionally, similar models parse text from left-to-right or right-to-left
   * However, BERT is able to parse text from *both directions* through a mask
   * BERT model involves masking 15% of the words or so, and the model then attempts to guess what these masked words are
   * Actually, of these masked words, 80% are actually masked, 10% are random words, and 10% are original words. Paper says that this improves model, but not exactly sure how
   * By using masked words, model is able to build better long-term relationships between the masked word and words *both* before and after
 * BERT also involves predicting the next sentence -- if the model is able to predict whether or not the next sentence is related to the current one, then it has a better understanding of the current context

Here's a passage on how it works more specifically:
> In technical terms, the prediction of the output words requires:
* Adding a classification layer on top of the encoder output.
* Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
* Calculating the probability of each word in the vocabulary with softmax.

## Week 3 (1/21-1/27)

## January 26, Saturday (1.5 hours)
- [x] Continue to read about [BERT](http://jalammar.github.io/illustrated-bert/)

## January 25, Friday (2 hours)
- [x] Team meeting with May and William
- [x] Start reading about Bidirectional Encoder Representations from Transformers (BERT) [here](https://github.com/google-research/bert) and [here](http://jalammar.github.io/illustrated-bert/)

[Meeting Notes](https://docs.google.com/document/d/1yAOxQqH1dNwA_2YFLM9-U_hp70FiFaU7J7PBDIG99CA/edit), continued from yesterday's meeting inputs

Key Takeaways from Meeting
 * William suggested we look into the following: QA / dialogue answering tasks, gender bias in BERT, and SQUAT dataset
 * Causal graph strategy is immensely difficult because solving the notion of causality is a Turing Award winning question
 * LSTMs may have some potential, but there the coreference resolution field is already highly saturated
 * For GLOMO, we can try analyzing gender bias in the affinity matrices using nodes
 * If we can identify bias in BERT, that would be great because it is state-of-the-art, and everybody is using it

* Yuxin and I will look more into BERT; Shirlyn and Andrew will take a look at affinity matrices
* From initial readings, BERT is pretty difficult to understand

### January 24, Thursday (3 hours)
- [x] Read about Google's [Sentence Level Encoder](https://arxiv.org/pdf/1803.11175.pdf)
- [x] Review [GLOMO](https://arxiv.org/pdf/1806.05662.pdf) and [Avoiding Discrimination through Causal Reasoning](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)
- [x] Prep [Meeting Inputs](https://docs.google.com/document/d/1yAOxQqH1dNwA_2YFLM9-U_hp70FiFaU7J7PBDIG99CA/edit?usp=sharing) for tomorrow

Key Takeaways from [Sentence Level Encoder](https://arxiv.org/pdf/1803.11175.pdf)
 * Acts basically like a word embedding but for sentences
 * Can be formed either through a transformer model or DAN model
   * Transformer model is *O(n^2)*, whereas DAN is *O(n)*, but generally transformer model will produce better results
 * Potential to apply this to LSTM strategy to debias by evaluating degree of gender bias in sentence embeddings (and therefore we can either inject new data points to balance it out or perhaps try Bolukbasi's method to remove gender bias)

Key Meeting Inputs
 * Three main strategies for debiasing
 * We have read a lot about causal graphs, and they seem promising, but it seems difficult to make a practical causal graph with a predictor that would suit our needs
 * Andrew worked on some WEAT tests for implicit hate speech detection, but no luck
 * Didn't find too much info for incorporating gender debiasing with LSTMs, but maybe sentence level embeddings may provide an avenue

### January 23, Wednesday (2.5 hours)
- [x] Attend NLP lab meeting -- discussion on [A new method of region embedding for text classification](http://research.baidu.com/Public/uploads/5acc1e230d179.pdf), and gave feedback to two members practicing AAAI spotlight talks
- [x] Team meeting to update each other on our individual progress -- things we did, obstacles, and things to work on by Friday; also discussed the papers we have read

### January 22, Tuesday (2.5 hours)
- [x] Skype call with Andrew, Yuxin, Shirlyn, and May -- mainly May helped us clarify questions we had from the papers
- [x] Read through [A new method of region embedding for text classification](http://research.baidu.com/Public/uploads/5acc1e230d179.pdf) for NLP lab meeting tomorrow

[Meeting Notes](https://docs.google.com/document/d/1Ohm3tSf7-dMn_A6Zyl55fVkSvuicqYVPwvMwqpQpZTQ/edit?usp=sharing)

### January 21, Monday (2 hours)
- [x] Read through [GLOMO](https://arxiv.org/pdf/1806.05662.pdf) paper on transfer learning
- [x] Skim through [Automatic Extraction of Causal Relations from Text using Linguistically
Informed Deep Neural Networks](http://www.aclweb.org/anthology/W18-5035)

Key Takeaways from [GLOMO](https://arxiv.org/pdf/1806.05662.pdf) paper on transfer learning
 * Usually transfer learning incorporates learning some kind of feature vector and applying that to other methods
   * Has found success with MNIST dataset
 * GLOMO adds another layer to the transfer learning process by taking a sentence and forming a T x T matrix, where *T* is the length of the sentence
   * The value at a given row and column index represents how much in common the ith and jth words are
   * Then you add multiple layers for multiple sentences, which then forms this 3D tensor
   * Users can take this and apply it to other tasks
   
* A little bit confused on [Automatic Extraction of Causal Relations from Text using Linguistically
Informed Deep Neural Networks](http://www.aclweb.org/anthology/W18-5035), will ask May more about it tomorrow

___

## Week 2 (1/14-1/20)

### January 20, Sunday (3.5 hours)
- [x] Review Caliskan's paper regarding [WEAT](http://science.sciencemag.org/content/356/6334/183)
- [x] Finish reading [Avoiding Discrimination through Causal Reasoning](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)

Key Takeaways from [Avoiding Discrimination through Causal Reasoning](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)
 * The paper is theoretical -- no results regarding how to actually build a causal graph
 * But if a directed, acyclic causal graph were to be built, then one could identify bias (in general, not just gender bias) through both **resolving** and **proxy** variables
   * A causal graph exhibits discrimination if a protected attribute, say gender, does *not* pass through a resolving variable
   * In other words, if a protected attribute like gender directly affects the prediction, then there is bias
 * There is also a notion of *proxy* variables, which are "clearly defined observable quantities that is significantly correlated with A (the protected attribute), yet in our view should not affect the prediction
   * Therefore, the paper proposes a way to remove the affect of the proxy variable on the prediction
   * A little unclear exactly how this works -- perhaps can clarify with team / May / William

From the conclusion:
> Key concepts of our conceptual framework are resolving variables and proxy variables that play a dual role in defining causal discrimination criteria. We develop a practical procedure to remove proxy discrimination given the structural equation model and analyze a similar approach for unresolved discrimination... Our framework is limited by the assumption that we can construct a valid causal graph

Key Takeaways from [WEAT](http://science.sciencemag.org/content/356/6334/183)
 * WEAT (Word Embedding Association Test)  is a method to show bias in word embeddings
 * Turns out that bias in word embeddings matches those found in humans (which are found using the Implicit Association Test)
 * The method for WEAT incorporates using two classes (say European-American names and African-American names) and two attributes (say pleasant and unpleasant)
   * Then see how similar one class is to one attribute compared to the other class, e.g. on average, do the word embeddings for African-American names have a closer association with "unpleasant?"

### January 19, Saturday (3 hours)
- [x] Play around with multiple linear regression on Python
- [x] Start reading [Avoiding Discrimination through Causal Reasoning](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)

### January 18, Friday (2.5 hours)
- [x] Meet with Andrew, May, and William to discuss research for this quarter
- [x] Look through some notable papers on information extraction from a [Github](https://github.com/thunlp/NREPapers) William recommended

Key Takeaways from Meeting
 * Using coreference resolution for debiasing seems like just an incremental improvement, not something that would lead the field or is groundbreaking
 * Causal graph debiasing seems interesting, William recommended we look at [GLOMO paper](https://arxiv.org/pdf/1806.05662.pdf), which can be useful for transfer learning
 * Look at WEAT if we are considering to look more into implicit hate speech detection

[Meeting Notes](https://docs.google.com/document/d/1q5CvZvB1Jf3I4NPYkFevJR6T6MFgK3GzD_zg39MgXzg/edit) (continued from Wednesday)

### January 16, Wednesday (2 hours)
- [x] Meet with May, Andrew, and Shirlyn to discuss potential paths for research
- [x] Read a little about long short-term memory models for machine learning

Key Takeaways from Meeting -- 3 possible ideas
 1. Debiasing inputs on the fly, automatically (using LSTMs)
   * Use LSTMs to know how biased the previous X number of inputs are
   * If the input data is too biased in one direction, inject data that would balance it out, possibly using coreference resolution and gender swapping
 2. Implicit hate speech tweet detection
   * Try to tease out gender bias from sentences while avoiding confounding variables
   * A lot of work has been in *explicit* hate speech detection, but so far implicit hate speech detection has been difficult
 3. Causal graph debiasing
   * Idea from [this paper](http://papers.nips.cc/paper/6668-avoiding-discrimination-through-causal-reasoning.pdf)
   * Strategy would be to build a causal graph and find a way to debias accordingly

* May says it would be difficult to collaborate with other departments simply because it is hard to find a consistent, reliable source who will work with us for the full quarter

[Meeting Notes](https://docs.google.com/document/d/1q5CvZvB1Jf3I4NPYkFevJR6T6MFgK3GzD_zg39MgXzg/edit?usp=sharing)

### January 15, Tuesday (1 hour)
- [x] Read through last quarter's research paper to get ideas for this quarter's research

___

## Week 1 (1/7-1/13)

### January 9, Monday (1 hour)
- [x] Write down potential ideas for this quarter's research
