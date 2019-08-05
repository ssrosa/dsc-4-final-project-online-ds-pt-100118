# A Perceptron of the Artist as a Young Man
### in which a neural network and I enjoy a book together

In James Joyce's *A Portrait of the Artist as a Young Man*, the author controls language to show his protagonist maturing over time. The language used from one chapter to the next is so different that a human reader notices the change immediately when reading. This project uses shallow classifiers and deep neural networks to attempt to get a machine to do the same thing: if presented with a sentence from the book, could a machine identify that sentence's chapter? 

## Contents
**index.ipynb**: A Jupyter notebook with the classifiers, analysis, and visualizations.
**joyce_functions.py**: A script with functions for organizing and interpreting the text as word vectors.
**nontechnical_slidedeck.pdf** A slide show for non-technical audiences, in this case journalists, who receive a gentle admonishment for their breathless reporting on the "intelligence" of book recommendation systems.
**chapter1.txt** through **chapter5.txt** The raw text of the novel taken from Projecct Gutenberg.

## Methods
Stop words were removed from the text and then words were turned into 300-dimension vectors via Word2Vec with Spacy. Then a 'sentence vector' was found for each sentence by finding the mean of the vectors of all the words in that sentence. Sentence vectors were used to train the classifiers.

## Obversations
I made some startling observations while exploring the data. I was curious to see if the placement of the word vectors in space would form any pattern I could discern, so I used principal component analysis to reduce the 300 dimensions of each vector down to 3. I plotted the 3D word vectors in space and found to my amazement that the words were clustered together in intuitive ways, e.g. one blob in the corner of the 3D plot contained mostly proper names. Another area contained words related to sports, another contained words related to sea-faring, &c.

## Results

Logistic regression models, with some hyperparamater tuning, achieved 46% accuracy.

Random forest classifiers did better, achieving 53% accuracy.

Neural networks achived 55% accuracy but dense networks could not do the job on their own. I had to add in regularization and a GRU layer for short-term memory. And the 300-dimension vectors did not feed into the model correctly so the text had to be reprocessed with Keras's own NLP preprocessing tools and then fed into an embedding layer.

The highest accuracy thus achieved by a classifier was 55% (compared to 20% random guessing.) This is not a bad result and shows that a good bit of the pattern in the language apparently translates well into a pattern in numerical vectors. But it is well below what we could expect from a human reader.