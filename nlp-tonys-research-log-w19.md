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

___

### January 15, Tuesday (1 hour)
- [x] Review last quarter's research paper to get ideas for this quarter's research

## Week 1 (1/7-1/13)

### January 9, Monday (1 hour)
- [x] Write down potential ideas for this quarter's research
