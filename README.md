# README - COSI 114b Part-of-Speech Tagging with Structured Perceptrons 

The code provided in this repository contains the solutions to the Part-of-Speech Tagging with Structured Perceptrons PA for COSI 114b - Fundamentals of Natural Language Processing II. 


## Installation and Execution 

Get the files from GitHub and in your terminal/console move into the project folder. Run the file with the solutions with the following line: 

``` bash 
python pos_tagger.py 
```

Doing the above will run the program using data given to the students in the instructions. The data is included in the compressed folders called ``` brown.zip ``` and ``` data_small.zip ```, so be sure to decompress the files before running the line above. 

Note: These instructions assume that the user has python downloaded and is able to run the ``` python ``` command in their terminal. If they don't, they can either set their device up to be able to run the command, or they can open the ``` pos_tagger.py ``` file in a separate application and run it through there. 


## Assignment Description 

The task is to implement a structured perceptron to perform part-of-speech tagging by completing the following functions: 

* ``` make_dicts(self, train_set) ``` - Use ``` self.tag_dict ``` and ``` self.word_dict ``` to translate between indices and either parts of speech or words, respectively. This function should, given the training set, fill in these dictionaries. You do not need to account for the start symbol ``` <S> ```, the stop symbol ``` </S> ```, or the unknown word ``` <UNK> ```. 

Note that although ``` / ``` is the separator between words and parts of speech, some words also contain ``` / ``` in the middle of the word. In these cases, it's the last ``` / ``` that separates the word from the part of speech.  

* ``` load_data(self, data_set) ``` - Given a folder of documents (training, development, or testing), returns a list of ``` sentence_ids ``` (noting that a document can contain multiple sentences), and dictionaries of ``` tag_lists ``` and ``` word_lists ``` such that: 
    * ``` tag_lists[sentence_id] ``` = list of part-of-speech tags in the sentence 
    * ``` word_lists[sentence_id] ``` = list of words in the sentence 

You can assign each sentence a ``` sentence_id ``` however you want, as long as they are distinct. It may be helpful to store the tags and words in terms of their indices, using ``` self.tag_dict ``` and ``` self.word_dict ``` to translate between them. If you come across an unknown word or tag (in the development or testing sets), you may use the ``` self.unk_index ``` as the index for that word or tag. 

* ``` viterbi(self, sentence) ``` - Implement the Viterbi algorithm. Specifically, for each ``` sentence ```, given as a list of indices, you should fill in two trellises, ``` v ``` (for ``` viterbi ```) and ``` backpointer ```. You can refer to the pseudo-code given in Figure 8.10 of the Jurafsky and Martin textbook. Although the Figure describes the Viterbi algorithm in the context of hidden Markov models, our procedure is essentially the same. Some notes: 
    * Remember that when working with structured perceptron scores, you should add the scores, rather than multiply the probabilities. 
    * Note that operations like + are Numpy *universal* functions, meaning that they automatically operate element-wise over arrays. This results in a substantial reduction in running time, compared with looping over each element of an array. As such, your ``` viterbi ``` implementation should not contain any for loops that range over states (for loops that range over steps are fine). 
    * To avoid unnecessary for loops, you can use broadcasting to your advantage. Briefly, broadcasting allows you to operate over arrays with different shapes. For example, to add matrices of shapes (*a*, 1) and (1, *b*), the single column of the first matrix is copied *b* times, to form a matrix of shape (*a*, *b*). Similarly to add matrices of shapes (*a*, *b*) and (1, *b*), the single row of the second matrix is copies *a* times. 
    * When performing integer array indexing, the result is an array of lower rank (number of dimensions). For example, if ``` v ``` is a matrix of shape (*a*, *b*), then ``` v[:, t-1] ``` is a vector of shape (*a*, ). Broadcasting to a matrix of rank 2, however, results in a matrix of shape (*a*, 1): our column becomes a row. To get a matrix of shape (*a*, 1), you can either use slice indexing instead, or use the ``` numpy.reshape ``` function. 
    * If you come across an unknown word, you should treat the emission scores for that word as 0. 
    * In the transition matrix, each row represents a previous tag, while each column represents a current tag. 
    * Finally, you do not have to return the path probability, just the backtrace path. 

* ``` train(self, train_set) ``` - Given a folder of training documents, this function fills in ``` self.tag_dict ``` and ``` self.word_dict ``` using the ``` make_dicts ``` function, loads the dataset using the ``` load_data ``` function, shuffles the data, and initializes the three weight arrays ``` self.initial ```, ``` self.transition ```, and ``` self.emission ```. Then, for each sentence, this function does the following: 
    * Use the Viterbi algorithm to compute the best tag sequence. 
    * If the correct sequence and predicted sequence are not equal, update the weights using the structured perceptron learning algorithm: increment the weights for features in the correct sequence, and decrement the weights for features in the predicted sequence. We will assume a constant learning rate $\eta = 1$. Here, simpler is better - no fancy Numpy tricks needed. 

* ``` test(self, dev_set) ``` - Given a folder of development (or testing) documents, returns a dictionary of ``` results ``` such that: 
    * ``` results[sentence_id]['correct'] ``` = correct sequence of tags 
    * ``` results[sentence_id]['predicted'] ``` = predicted sequence of tags

* ``` evaluate(self, results) ``` - This function should return the overall accuracy (number of words correctly tagged / total number of words). You don't have to calculate precision, recall, or F1 score. You should be able to get an accuracy of about 85% on the development set. 

Hint: It took about **9 minutes** to train and test the model on the full dataset. Your milage may vary, depending on your computer. 
