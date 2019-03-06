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


## Week 9 (3/4-3/10)

### March 4, Monday (14 hours)
 - [x] Finish 8-page and 4-page papers!
 - [x] What a crazy journey
 
Crown jewel from 4-page paper:

![crown jewel](https://user-images.githubusercontent.com/36688734/53865578-613bb480-3fa4-11e9-9635-169149939b35.png)

Relation graph:
![relations](https://user-images.githubusercontent.com/36688734/53865676-9942f780-3fa4-11e9-8e5a-b52f089c27d4.png)


## Week 8 (2/25-3/3)

### March 3, Sunday (6 hours)
 - [x] Meet with Mai and Andrew to discuss statistical testing for OpenIE  
 - [x] Read about another [gender bias evaluation testset](https://arxiv.org/abs/1810.05201)
 - [x] Reread past papers for survey paper
 - [x] Continue to make significant edits to 8-page paper

### March 2, Saturday (4.5 hours)
 - [x] Read three research papers: [quantifying gender bias](https://arxiv.org/pdf/1607.03895.pdf),  [glass ceiling in NLP](http://aclweb.org/anthology/D18-1301), [differential responses to gender](https://web.stanford.edu/~jurafsky/pubs/rtgender.pdf)
 - [x] Continue to revise both 8-page paper
 - [x] Make suggestions to 4-page paper

### March 1, Friday (5.5 hours)
- [x] Meet with team, William, and Mai to discuss final steps for paper 
- [x] Make significant edits to 8-page paper
- [x] Continue working on graphs for 4-page paper

### February 28, Thursday (5 hours)
- [x] Review Open IE [github](https://github.com/dair-iitd/OpenIE-standalone) and go over [BONIE](http://www.cse.iitd.ac.in/~mausam/papers/acl17.pdf)
- [x] Read about OLLIE [here](https://homes.cs.washington.edu/~mausam/papers/emnlp12a.pdf) and [here](https://github.com/schmmd/ollie/blob/master/README.md)
- [x] Go over current state of both research papers
- [x] Identify discrepancies in original text data, Open IE outputs, and our occupation gender-swapped outputs
- [x] Continue working on visualizing data from Open IE graph extraction

Occupation Frequency Results from Gender-swapping the dataset in Open IE 5.0 (Female)
![Open IE 5.0 (F)](https://user-images.githubusercontent.com/36688734/53625279-7f29a380-3bb7-11e9-8d62-5e255dc06877.png)

Occupation Frequency Results from Gender-swapping the dataset in Open IE 5.0 (Male)
![Open IE 5.0 (F)](https://user-images.githubusercontent.com/36688734/53625553-450cd180-3bb8-11e9-917c-a938961faafb.png)

Occupation Frequency Results from Gender-swapping the dataset in Stanford Open IE (Female)
![Stanford (F)](https://user-images.githubusercontent.com/36688734/53625764-d3815300-3bb8-11e9-84f9-ca21cb48613e.png)

Occupation Frequency Results from Gender-swapping the dataset in Stanford Open IE (Male)
![Stanford (M)](https://user-images.githubusercontent.com/36688734/53625823-fe6ba700-3bb8-11e9-9ca7-88e93cb153ad.png)


### February 27, Wednesday (5 hours)
- [x] Learn a bit of Excel / VBA / Google Sheets / Google Scripts to visualize some of our data
- [x] Gender bias in Open IE! Especially the Stanford one: sentences that were gender-swapped from female to male had more than **4 times** the amount of relations for the same dataset for males without gender-swapping
- [x] Read about [CALMIE](http://www.cse.iitd.ac.in/~mausam/papers/coling18.pdf)

Gender-swapping the dataset in Open IE 5.0 (Female)
![Open IE 5.0 (M)](https://user-images.githubusercontent.com/36688734/53615362-6064e600-3b92-11e9-930c-fc5808a1fb77.png)


### February 25, Monday (3 hours)
- [x] Meet with team and May to discuss things we should address for our 8-page paper and how to move forward with our 4-page one
- [x] Look into results of gender-swapped Open IE for discrepancies in occupations; my idea is that if Open IE is unbiased, then counts of occupations with each gender should not differ

## Week 7 (2/18-2/24)

### February 24, Sunday (1.5 hours)
- [x] Look a bit into our gender-swapped Open IE results
- [x] Read some more about [Open IE Stanford](https://nlp.stanford.edu/software/openie.html)

### February 23, Saturday (2 hours)
- [x] Learn the basics of how [Open IE 5.0 (Standalone)](https://github.com/dair-iitd/OpenIE-standalone) works
- [x] Go over our paper from last quarter as well as reviewer comments

Key Takeaways from Reviewers' Comments:
 * Mixed reviews: one reviewer commented that our conclusion / future directions contained some "unwarranted ideas," but the other two mentioned that those were interesting points about the future direction of the field
 * They liked how we categorize things, e.g. types of representaton bias, inference / retraining
 * Paper was mostly comprehensive, but one person said we could look into a few other papers for additional insight
 * We mixed up FPED / FNED a bit in section 2.2
 * I think we need to clearly define our audience in the introduction so that the reviewers have clear expectations

### February 22, Friday (4.5 hours)
- [x] Meet with William, May, and team to discuss results and future directions
- [x] Discuss with team potential strategies for identifying bias in OpenIE graph extractions
- [x] Rejected from NAACL -- almost had it though! William said that we were borderline acceptance and that around only 12% of the papers from his lab were accepted. May also told us to keep our chin up. 

William believes that there is potential for a short 4-page paper that we can submit to ACL by March 4. In his opinion, graph extractions with OpenIE seem to be the most promising idea because this is a relatively novel direction. Conversely, BERT, although state-of-the-art, would attract more competition especially for the topic word embeddings.

Since graph extraction is now our most promising direction, we came up with two main methods for gender bias identification:
1. Gender-swapping graph
  * We plan to gender-swap all he and she pronouns in the 30,000ish triples. When we visualize the graph, only the root nodes (he and she) should change. This is because if OpenIE is unbiased, it will only look at sentence structure to form triples as opposed to semantic meaning of words, which can introduce bias. 
  * Replacing he and she and vice versa will not make an impact on sentence structure, so all the occupations that were previously associated with "he" should all be associated with "she" after gender-swapping.
  * If there is bias, we can measure the degree to which the "occupation tree" has changed for each gender to quantitatively analyze bias.
2. Identifying bias in datasets
  * It is difficult to analyze bias directly in the dataset without verifying that OpenIE is unbiased because it becomes difficult to see whether the bias is from the dataset or OpenIE.
  * If we find that OpenIE is unbiased, we can say that a large number of the associations "male physicians", for example, is actually from the dataset and not because OpenIE is making incorrect triples.
We can use this to identify bias in many documents, which can be useful for future researchers if they want to better understand their dataset.

![Week 7 ERSP](https://user-images.githubusercontent.com/36688734/53280121-935b3580-36cb-11e9-8a1f-0cbbf300b125.png)

### February 21, Thursday (6.5 hours)
- [x] Meet with Yuxin to discuss multi-head attention in BERT
- [x] Discovered some extent of gender bias via the BERT visualization tool, but not fully conclusive evidence
- [x] Read Jay Alammar's blog on [encoders](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) and Lillian Weng's post about [attention mechanisms](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [x] Combine the resources I went over to create a script that would extract and plot contextualized word embeddings
- [x] Used tSNE to reduce the 758-dimensional word embedding vectors to two dimensions

Here are the results:

Base form
![embedding_base](https://user-images.githubusercontent.com/36688734/53279876-1038e000-36c9-11e9-89d8-ce693f9838cc.png)

Plotting the twenty most gender-biased occupations. Pink circles means that it is female-biased; blue circles means it is male-biased.
![embedding_occupations](https://user-images.githubusercontent.com/36688734/53279844-c18b4600-36c8-11e9-9de4-5c143ecafb13.png)

Plotting gender-biased words from two news articles. Pink circles means that the word appeared in a female-biased context; blue circles means that the word appeared in a male-biased context
![embedding_article](https://user-images.githubusercontent.com/36688734/53279897-4b3b1380-36c9-11e9-919b-18a88a65ec63.png)

I notice a general trend in which data points seem to form a circle with male datapoints on the exterior and female points on the interior. This is indicative of gender bias because in a gender-neutral system, male datapoints and female datapoints should be intertwined. 

I got inspiration from the [Deep Learning Book](https://www.deeplearningbook.org/) for a method to easily classify these points using polar coordinates as opposed to cartesian coordinates. Perhaps this can pave a way for debiasing in the future.
![polar coordinates](https://user-images.githubusercontent.com/36688734/53280076-3f505100-36cb-11e9-8e43-01eee62ee5cd.png)

### February 19, Tuesday (2.5 hours)
- [x] Review and reread [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)
- [x] Go over [BERT as a service] again, look into how we might be able to utilize word embeddings
- [x] Learn more about [tSNE] for dimensionality reduction

### February 18, Monday (1 hour)
- [x] Read more about [deconstructing BERT](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)
- [x] Explain how to use AWS EC2 to Yuxin

## Week 6 (2/11-2/17)

### February 17, Sunday (2.5 hours)
- [x] Look into the [BERT visualization tool](https://colab.research.google.com/drive/1Nlhh2vwlQdKleNMqpmLDBsAwrv_7NnrB) that William sent us
- [x] Read about [distilling BERT](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77) to gain a better understanding of how the tool works

![BERT visualization 1](https://user-images.githubusercontent.com/36688734/53280102-79215780-36cb-11e9-8d41-263ffa82c2f8.png)

![BERT visualization 2](https://user-images.githubusercontent.com/36688734/53279613-b8997500-36c6-11e9-954b-90490983ebdd.png)

### February 15, Friday (3.5 hours)
- [x] Meet with William, May and team -- [meeting outputs](https://docs.google.com/document/d/1butKQCg62in1NO2MQmuOrAq32IE93FU_yglA2zW3J1o/edit?usp=sharing)
- [x] Continue working on occupation.py and its variants

![Week 6 ERSP](https://user-images.githubusercontent.com/36688734/53279810-75d89c80-36c8-11e9-933d-9f989d9bea90.png)

### February 14, Thursday (7 hours)
- [x] Andrew was able to set up a virtual server following the instructions on [Bert as a Service](https://github.com/hanxiao/bert-as-service)
- [x] Created a bunch of scripts to try and test for gender bias, occupation.py seems like the one with the most promising results

How occupation.py works
 * I found that BERT outputs sentence embeddings, which I believe should represent the semantic representation of a sentence
   * Seems similar to Google's [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175), potential room for future direction
 * I take sentence pairs such as "she is a doctor", "he is a doctor" and compare cosine similarity between the two sentences
   * I also replace the occupation "doctor" with 30 total occupations from three categories (10 in each category): female-biased, male-biased, and gender neutral occupations
   * Theoretically, for any given sentence, the cosine similarity should be decently high because the only differences between the two sentences are "she" vs "he"
   * However, I hypothesized that cosine similarity will be lower for sentences with gender-biased occupations compared to sentences with gender-neutral ones because if BERT is responsive to gender bias, there would not be much of a difference between "she is a friend" and "he is a friend"

Below are the results from occupation.py -- green pluses represent gender neutral occupations, red circles represent female-biased occupations, blue circles represent male-biased occupations
x-axis measures cosine similarity

Results generally follow my hypothesis that gender-neutral occupations lead to higher cosine similarity.

![occupation.py](https://user-images.githubusercontent.com/36688734/53152649-f5e8f000-356a-11e9-8076-67ca9bad5bb1.png)

I also tried to control for sentence length affecting gender bias by extending sentence length while maintaining mostly the same semantic meaning. The base sentence for occupation_longer.py is "she / he aspires to be ___ in the future"

Results are more concentrated in this one as all occupations have high cosine similarity (scale in graph is from 0.8 to 1). This suggests that the sentence length / phrasing does affect cosine similarity, at least to some degree. Perhaps it's because there are more words in common

![occupation_longer.py](https://user-images.githubusercontent.com/36688734/53152768-4fe9b580-356b-11e9-8d7e-ffa4de51615a.png)

Also considered shorter sentences with only 2 words but it doesn't produce meaningful results as cosine similarity is only -1 or 1 (the circles and pluses are all stacked on top of each other)

![occupation_shorter](https://user-images.githubusercontent.com/36688734/53153316-d5219a00-356c-11e9-8207-77777128f45b.png)

### February 13, Wednesday (1 hour)
- [x] Reinstalled Anaconda and Jupyter to try and run programs on local machine
- [x] Look into [Bert as a Service](https://github.com/hanxiao/bert-as-service) -- it seems like a promising avenue to finally be able to work with BERT

### February 12, Tuesday (2.5 hour)
- [x] Consult with Yuxin how to tackle visualizing BERT embeddings
- [x] Read about the [Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) and more about [BERT](https://www.kdnuggets.com/2018/12/bert-sota-nlp-model-explained.html)
- [x] Tried to run Yuxin's code to set up BERT, but had environment issues

### February 11, Monday (2.5 hours)
- [x] Work with Andrew to get TensorFlow and TensorFlow-hub set up on AWS EC2
- [x] Successfully run some test code for spam detection classifier
- [x] Briefly look into how we can use this for BERT

From Google's Blog on the Transformer
> More specifically, to compute the next representation for a given word - “bank” for example - the Transformer compares it to every other word in the sentence. The result of these comparisons is an attention score for every other word in the sentence. These attention scores determine how much each of the other words should contribute to the next representation of “bank”. In the example, the disambiguating “river” could receive a high attention score when computing a new representation for “bank”. The attention scores are then used as weights for a weighted average of all words’ representations which is fed into a fully-connected network to generate a new representation for “bank”, reflecting that the sentence is talking about a river bank... Beyond computational performance and higher accuracy, another intriguing aspect of the Transformer is that we can visualize what other parts of a sentence the network attends to when processing or translating a given word, thus gaining insights into how information travels through the network.

![Transformer](https://user-images.githubusercontent.com/36688734/52675904-21ba0500-2edd-11e9-97c4-0a5cb29ff997.png)

## Week 5 (2/4-2/10)

### February 9, Saturday (4 hours)
- [x] Read about different types of [word embedding models](https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec) and the [attention mechanism](https://skymind.ai/wiki/attention-mechanism-memory-network)
- [x] Learn how to use AWS servers
- [x] Spent a lot of time trying to run [test code](http://hunterheidenreich.com/blog/elmo-word-vectors-in-keras/) on local machine and on AWS machine -- fixed some issues from yesterday but new tensorflow problems are still cropping up

### February 8, Friday (3 hours)
- [x] Attend ERSP meeting
- [x] 1-on-1 with May
- [x] Tried running [code](https://github.com/google-research/bert) for BERT, still running into issues with packages -- will continue to look into this

[Meeting Outputs](https://docs.google.com/document/d/1Rlk-RX5eZ7EK8nIAAn5iFOtBlJ8KvM_7hzQSnD_ONXk/edit?usp=sharing)

![Week 5_ERSP](https://user-images.githubusercontent.com/36688734/52473520-d903f400-2b4a-11e9-85aa-a222728d54c7.png)

### February 7, Thursday (5 hours)
- [x] Read [Google's blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) and [Medium article](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) on BERT
- [x] Sift through SQuAD 2.0 and 1.1 [dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [x] Watch lecture on [recurrent neural networks](https://www.youtube.com/watch?v=UNmqTiOnRfg)
- [x] Set up [virtual environment](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46) for Python on Jupyter Notebook
- [x] Read about BERT fine-tuning with [Google's Cloud GPU](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb#scrollTo=191zq3ZErihP)
- [x] Tried running some test code on spam classification but was running into issues with tensorflow

Key Takeaways
 * To my understanding, BERT outputs pre-trained contextualized word embeddings that can be fine-tuned to your task at hand, e.g. QA, using token, segment, and position embeddings to represent the input
   * BERT is deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus because it uses the concept of language modeling and masks to make use of bidirectional representation
 * We can try to analyze BERT for gender bias for tasks on which it is fine-tuned, such as question answering
   * The current state-of-the-art QA dataset is the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset by Stanford
   * However, I did not find that the dataset that is geared specifically towards gender and the answers are all very factual, 1-word, which makes it difficult to analyze outside the context of accuracy
   * I think to tease out bias we should try to use BERT, train with [Cloud GPU](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb#scrollTo=191zq3ZErihP) to fine-tune on some QA dataset (still looking for one that would fit our needs) relating to gender and evaluate results

   

[Meeting Inputs](https://docs.google.com/document/d/1X6ya88bv5dJSX9SDihWyKOzd4EmcFa-fFpbzBH7V2QM/edit?usp=sharing) for tomorrow's meeting 

### February 5, Tuesday (0.5 hours)
- [x] Look through a bit of the [BERT github](https://github.com/google-research/bert)

### February 4, Monday (1.5 hours)
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

![Week 4_ERSP](https://user-images.githubusercontent.com/36688734/52470503-2a0fea00-2b43-11e9-9e04-0c5849ad9392.png)

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
