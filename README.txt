Overview:

This assignment has two tasks, implementing the forward calculation of RNN, and language modeling with RNN. 


In the first task, you will implement two types of RNN networks, a basic one and a GRU. Your implementation needs to match tensorflow calculations. Then you will test long-term dependency and vanishing gradient of two RNN networks on a random task. 


The second task is taken from the Stanford 224d course. In this task, you will implement an RNN model for language modeling. An input sequence to the model is a sentence consisting of word tokens, and the fitting target at every step is the next word in the sentence. The goal is to improve the predictive accuracy of the model, which is measured by the perplexity. The model can also generate text, and the generated text is also an evidence whether the model captures the pattern of the language. Please check the section 3 of the handout from the course (http://cs224d.stanford.edu/assignment2/assignment2.pdf). Please note the difference of requirements: we are allowed to use tensorflow RNN cells while that course does not. 

Points breakdown: 

25% Task 1 (see the point breakdown in the notebook)
30% Implementing the RNN model for language modeling  
30% Model tuning 
10% Achieve a validation perplexity value less than 175
5% Generate reasonable text 
