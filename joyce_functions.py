import copy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #For viewing 3d pca feature plots
import seaborn as sns

#For NLP preprocessing
from nltk.corpus import stopwords
from nltk import word_tokenize
import string #for removing punctuation from text
import en_core_web_md #parrish
import spacy #parrish 

# MODELING:
from sklearn.decomposition import PCA #For squashing word vector means
from sklearn.preprocessing import StandardScaler #For normalizing data
#To divide data when testing a trained model
from sklearn.model_selection import train_test_split 
#To build regular logistic regression models
from sklearn.linear_model import LogisticRegression
# ENSEMBLE METHODS:
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# ASSESSING MODELS:
#To assess accuracy of logistic regression or decision trees
from sklearn.metrics import confusion_matrix 
#To iteratively append labels to cells in a confusion matrix
import itertools 
#To get accuracy, precision, recall, and F1 score (weighted accuracy) of a given confusion matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#To view the accuracy metrics for a given confusion matrix
from sklearn.metrics import classification_report

#For neural network preprocessing
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, GlobalMaxPool1D, GlobalMaxPool2D, GRU
from keras.models import Model
from keras import models, initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence

nlp = spacy.load('en_core_web_md')

def stop(word):
    '''
    Returns a token if its survives stoppage.
    Parameters:
    word (str)
    '''
    stop_list = stopwords.words('english')
    stop_list += list(string.punctuation)
    stop_list += ['could', 'like', 'one', 'said', 'would', '\'s', '\'ll']
    
    if word.isalpha():
        if word.lower() not in stop_list:
            return word.lower()  

# FOR CONSTRUCTING DFS OF VECTORS ###########################
def word_vec_df(chapter, label):
    '''
    Constructs a df to track words in a chapter, each word's token,
    and each token's vector.
    Parameters:
    chapter (spacy.tokens.doc.Doc) The text of a chapter.
    label (int) The chapter number.
    Returns:
    df (pd.DataFrame)
    '''
    #Token for each word in the text that survives stoppage  
    #Only need unique tokens, so we take the set
    tokens = list(set([stop(t) for t in word_tokenize(chapter.text) if stop(t)]))
    
    #Vector for each token
    vectors = [nlp.vocab[t].vector for t in tokens]
    
    #Instantiate new df
    df = pd.DataFrame()
    #Build df columns from above lists
    df['token'] = tokens
    df['vector'] = vectors
    df['label'] = label
    
    return df

def sent_vec_df(chapter, labels):
    '''
    Consructs a df to track a chapter's sentences, each sentence's tokens,
    some stats on the tokens, each token's vector, and the 
    'mean vector' for each sentence's vectors.
   
    Parameters:
    chapter (spacy.tokens.doc.Doc) The text of a chapter.
    label (int) The chapter number.
    
    Returns:
    df (pd.DataFrame)
    '''
    #Turn spacy doc into lists of sentences as strings
    sentences = [str(s) for s in list(chapter.sents)]
    
    #Number of characters in each sentence
    char_counts = [len(s) for s in sentences]
   
    #The tokens for each sentence
    tokenized_sentences = [[stop(t) for t in word_tokenize(sentence) if stop(t)] \
                               for sentence in sentences]
    
    #Average length of token in each sentence
    mean_token_len = [np.mean([len(t) for t in tokens]
                             ) for tokens in tokenized_sentences]
    #Number of tokens in each sentence
    #(Those with 0 will be dropped later)
    token_counts = [len(t) for t in tokenized_sentences]
                               
    #The vector for each token
    vectorized_tokens = [[nlp.vocab[t].vector for t in tokenized_sent] \
                             for tokenized_sent in tokenized_sentences]
    
    #The 'mean vector' for each sentence's vectors
    #(Each mean vector is itself a 300-dimension vector like the originals)
    mean_vectors = [np.mean([v for v in vector_list], 
                           axis = 0
                          ) for vector_list in vectorized_tokens]
    
    #Instantiate new df
    df = pd.DataFrame()
    #Build df columns from above lists
    df['sentence'] = sentences
    df['char_count'] = char_counts
    df['token'] = tokenized_sentences
    df['mean_token_len'] = mean_token_len
    df['token_count'] = token_counts
    df['vectors'] = vectorized_tokens
    df['vector'] = mean_vectors
    df['label'] = labels
    
    #Drop rows which don't have any tokens
    df = df[df['token_count'] != 0]
    
    return df

def appears_in(text, chap, label):
    '''
    Gets words that are unique to a chapter. Checks a chapter's words
    against the words in the rest of the text.
   
    Parameters:
    text (pd.DataFrame) The full text. (Requires either a sv_df or wv_df.)
    chap (pd.DataFrame) Just this chapter.
    label (int) This chapter's number.
    
    Return:
    df (pd.DataFrame)
    '''
    #List of words from other chapters
    other_chapters = text[text['label'] != label]['token'].values
    
    #List of words from this chapter
    this_chapter = chap['token'].values
    
    #List of bools for words unique to this chapter
    unique_words = [True if v not in other_chapters else False \
                     for v in this_chapter]
    df = chap[unique_words]
    return df

# PCA functions ##############################################################
#vec_to_col
def vec_to_col(vector_arr):
    '''
    Turns an array of vectors into columns for each vector.
    Use df['vector'].values to pass in the array of vectors.
    It gets turned into a 300-wide array with as many rows as rows in the df.
    '''
    #Make new 2d array
    data = np.array(list(vector_arr))
    
    #Make list of column names
    #Assumes all vectors have same dimensions
    columns = [str(i) for i in range(data.shape[1])]
    
    #Instantiate new df to hold vector dimensions as columns
    vector_cols = pd.DataFrame(data = data, columns = columns)

    return vector_cols

def do_pca(df, normalize = False, components = 3):    
    '''
    
    '''
    #Turn vectors into columns.
    if normalize:
        vector_cols = StandardScaler().fit_transform(vec_to_col(df['vector'].values))
    else:
        vector_cols = vec_to_col(df['vector'].values)
        
    #Get principal components from features with PCA object
    pca = PCA(n_components = components)
    principalComponents = pca.fit_transform(vector_cols)

    # Create a new DataFrame for principal components 
    columns = ['PC' + str(i + 1) for i in range(principalComponents.shape[1])]
    pca_df = pd.DataFrame(data = principalComponents, columns = columns)
   
     #Add other features back
    pca_df['token'] = df['token'].values
    pca_df['vector'] = df['vector'].values
    pca_df['label'] = df['label'].values
    
    return pca_df

def get_xyz(xyz):
    '''
    Helper function for plotting.
    Parameters:
    xyz (list) List with 6 integers.
    Returns:
    6 integers in variables
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = xyz[0], xyz[1], xyz[2], xyz[3], xyz[4], xyz[5]
    return xmin, xmax, ymin, ymax, zmin, zmax

def plot_pca(df, plot_list, 
             xyz = None,
             figsize = (12, 14),
             azim = 60, 
             elev = 30,
             alpha = None,
             prismXYZ = None,
             legend = False,
             title = None):
    '''
    Draws a flexible figure of subplots, with as many as are listed
    in the plot list of dictionaries. Each subplot can have one or more
    scatters drawn together depending on how many key/val pairs are in
    the dict.
    Parameters:
    df (pd.DataFrame) Needs columns for 'PC1', 'PC2', and 'PC3.'
    plot_list (list) List of dictionaries. Each dict is a subplot with as many 
    scatters as there are key/val pairs in the dict. E.g. for one plot with one scatter,
    use a list with 1 dict with 1 key/val pair.
    xyz (list) List of limits for the three dimensions.
    figsize (tuple) Size of figure.
    azim (int): Horizontal view of the plot.
    elev (int): Vertical view of the plot.
    alpha (float): Value between 0 and 1. Transparency of scatters.
    prismXYZ (dict): Dict with prism names as keys and dimension lists as values.
    legend (bool): True if a legend should be drawn.
    title (str): Title of plot (only really if just one plot being shown.)
    
    Returns: 
    Draws a plot.
    '''
    #Get xyz limits for the plots.
    if xyz:
        xmin, xmax, ymin, ymax, zmin, zmax = get_xyz(xyz)
    
    #Get right number of rows whether length is even or odd.
    #Needed this instead of just rounding because of 'banker's rounding.'
    rows = (len(plot_list) // 2) + (len(plot_list) % 2)

    #Set columns. 1 if 1 plot, else 2.
    if len(plot_list) == 1:
        cols = 1
    else:
        cols = 2
    #Draw figure to hold all plots
    fig = plt.figure(figsize = figsize)    
    
    #Draw a subplot for each dict in the plot list.
    #If just one subplot should be drawn, put just one dict in the list.
    #The plot will fill the figure.
    for i in range(len(plot_list)): 
        ax = fig.add_subplot(rows, cols, (i + 1), 
                             projection='3d', 
                             azim = azim, 
                             elev = elev)
        if xyz: 
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
        
        #Draw each scatter within a subplot. 1 for each key/val pair in the dict.
        #(Allows for each subplot to show one or more scatters.)
        #together.
        for label, color in  plot_list[i].items():
            #Pull the data from the df corresponding to each label
            pca = df[df['label'] == label]
            #Draw a scatter for the data from each label
            ax.scatter(xs = pca['PC1'], 
                       ys = pca['PC2'], 
                       zs = pca['PC3'],
                       c = color,
                       alpha = alpha,
                       label = ('Chapter ' + str(label)))
        
        #Draw prisms
        if prismXYZ:
            #Draw each prism in list
            #Leaving prismXYZ as a dict for the moment in case I want to use labels
            for label, xyz in prismXYZ.items():
                draw_prism(xyz, 'black')
        
            
        #Label axes for each subplot
        ax.set_xlabel('x, PC1')
        ax.set_ylabel('y, PC2')
        ax.set_zlabel('z, PC3')
        
        #Set legend for each subplot
        if legend:
            plt.legend()
        #Set title for each subplot
        if not title:
            plt.title('Chapters: ' + str(list(plot_list[i].keys())))
        else:
            plt.title(title)
    plt.show()
    
def isolate(df, xyz):
    '''
    Parameters:
    df (pd.DataFrame) Needs a 'label' column to work. Needs PC1, 2, and 3 columns.
    xyz (list) List of limits for PC1, 2, and 3.
   
    Returns:
    blob_df (pd.DataFrame) Section of the pca_df within the 
    spatial limits set by the parameters. The 'blob' within 
    a 'cube' on a 3d plot.
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = get_xyz(xyz)
    
    blob = df[
                    ((df['PC1'] > xmin) & (df['PC1'] < xmax )) & \
                    ((df['PC2'] > ymin) & (df['PC2'] < ymax )) & \
                    ((df['PC3'] > zmin) & (df['PC3'] < zmax ))
                ]
    return blob

def draw_prism(xyz, color):
    '''
    When called within plot_pca, draws a prism in a 3d plot.
    Works well with isolate() to visualize the isolated tokens.
    '''
    #1 .   2 .   3 .   4 .   5 .  6
    xmin, xmax, ymin, ymax, zmin, zmax = get_xyz(xyz)
    c = color
    plt.plot([], [], [], color = c)
    
    kwargs = {'linewidth': 4}
    
    #I tried a lot of ways of consolidating these loops further
    #but decided to move on.
    #x limits at zmin
    for x in [xmin, xmax]:
        plt.plot([x, x], [ymin, ymax], [zmin, zmin], color = c, **kwargs)

    #y limiits at zmin
    for y in [ymin, ymax]:
        plt.plot([xmin, xmax], [y, y], [zmin, zmin], color = c, **kwargs)

    #x limits at zmax
    for x in [xmin, xmax]:
        plt.plot([x, x,], [ymin, ymax,], [zmax, zmax], color = c, **kwargs)

    #y limits at zmax
    for y in [ymin, ymax]:
        plt.plot([xmin, xmax], [y, y], [zmax, zmax], color = c, **kwargs)

    #z limits
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            plt.plot([x, x], [y, y], [zmin, zmax], color = c, **kwargs)
            
    #Draw label
    #ax.text(xmax, ymax, zmin, s = label)
##############################################################
def plot_conf_matrix(cm, classes, normalize=False, 
                          title='Confusion Matrix', cmap=plt.cm.Blues):
    '''
    Draws a heat map to show true positives, false positives, &c
    for given predicted y values vs actual y values.
    Parameters:
    cm (np.array) The confusion matrix for a model's predictions.
    classes (list) Names of classes/categories.
    normalize (bool) Whether to normalize the numbers in the matrix.
    Returns:
    Visualized heat map of the confusion matrix.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# MODELING ###############################
def prepare(df,
            pca_components = None,
            normalize_vector = False,
            features_to_norm = None, 
            other_features = None,
            y_to_dummies = False
           ):
    '''
    Pipeline for preparing data for modeling.
    Parameters:
    df (pd.DataFrame) Sentence vector DataFrame.
    pca_components (int) Number of principal components to extract from vectors.
    normalize_vector (bool) Whether to normalize the vector columns.
    features_to_norm (list) Features (other than 'vector') to include in the model.
    other_features (list) Features to include without normalizing.
    y_to_dummies (bool) Whether to get dummies for the labels. (Needed for neural network.)
    
    '''
    #2 Use PCA to get a few columns from the hundreds of vector columns
    if pca_components:
        #1 Turn vector into columns
        vector_cols = vec_to_col(df['vector'].values)
        
        pca = PCA(n_components = pca_components)
            #Returns an array with as many columns as you chose components
        if normalize_vector:
            principalComponents = pca.fit_transform(StandardScaler().fit_transform(vector_cols))
        else:
            principalComponents = pca.fit_transform(vector_cols)   
            #Create a new DataFrame for principal components 
        columns = ['PC' + str(i + 1) for i in range(principalComponents.shape[1])]
        pca_df = pd.DataFrame(data = principalComponents, columns = columns)
        # Combine features into df
        X = copy.deepcopy(pca_df)
    else:
        X = pd.DataFrame()

        #If features other than the vectors are to be normalized
    if features_to_norm:
        normed = StandardScaler().fit_transform(df[features_to_norm])
        normed_df = pd.DataFrame(data = normed, columns = features_to_norm)
        X = pd.concat([X, normed_df], axis = 1)
    
        #If any other features, not normalized, are to be included
    if other_features:
        X = pd.concat([X, df[other_features]], axis = 1)
    
    #5 Get target
    if y_to_dummies:
        y = pd.get_dummies(df['label'].values)
    else:
        y = df['label']
    
    #[OPTIONAL] Adjust sizes of chapters  ####
    
    #6 Do train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
    
    return X_train, X_test, y_train, y_test

#LOGISTIC REGRESSION
def lr(X_train, X_test, y_train, y_test,
      report = True):
    '''
    Pipeline for performing logistic regression.
    Parameters:
    (X_train...) (arr) The features and the labels as returned by the 
    'prepare' pipeline.
    report (bool) Whether to print a scores report.
    '''
    #Instantiate model
    logreg = LogisticRegression(fit_intercept=False, C=1e16)
    
    # Fit the training data to the model
    logreg.fit(X_train, y_train)

    # Generate predicted values for y to compare to real values
    y_hat_train = logreg.predict(X_train)
    #Now generate predicted values for the test data to compare
    y_hat_test = logreg.predict(X_test)
    
    #Print scores
    target_names = [str(c) for c in logreg.classes_]
    if report:
        print('LogReg, training set:')
        print(classification_report(y_train, y_hat_train, target_names = target_names))
        print('LogReg, testing set:')
        print(classification_report(y_test, y_hat_test, target_names = target_names))
    
    return logreg.score(X_train, y_train), logreg.score(X_test, y_test)

#RANDOM FOREST
def rf(X_train, X_test, y_train, y_test,
       n_estimators=175,
       max_depth = 30,
       report = True):
    '''
    Pipeline for running a random forest.
    Parameters:
    X_train... (arr) The features and the labels as returned by the 
    'prepare' pipeline.
    n_estimators (int) Hyperparameter to adjust.
    max_depth (int) Hyperparameter to adjust.
    report (bool) Whether to print a scores report.
    '''
    #Instantiate model
    forest = RandomForestClassifier(n_estimators = n_estimators, 
                                max_depth = max_depth, 
                                bootstrap = True
                               )
    
    # Fit the training data to the model
    forest.fit(X_train, y_train)

    # Generate predicted values for y to compare to real values
    y_hat_train = forest.predict(X_train)
    #Now generate predicted values for the test data to compare
    y_hat_test = forest.predict(X_test)
    
    #Print scores
    target_names = [str(c) for c in forest.classes_]
    if report:
        print('Random forest, training set:')
        print(classification_report(y_train, y_hat_train, target_names = target_names))
        print('Random forest, testing set:')
        print(classification_report(y_test, y_hat_test, target_names = target_names))
    
    return y_hat_train, y_hat_test, forest.score(X_train, y_train), \
            forest.score(X_test, y_test)

#NEURAL NETWORK
def plot_nn(model_dict):
    '''
    Plots loss and accuracy of a neural network side-by-side .
    Parameters:
    model_dict (model.history.history) Dictionary of values for loss and 
    accuracy of training and validation sets.
    '''
    loss_values = model_dict['loss']
    val_loss_values = model_dict['val_loss']
    acc_values = model_dict['acc'] 
    val_acc_values = model_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    fig = plt.figure(figsize = (14, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, loss_values, 'g', label='Training loss')
    ax1.plot(epochs, val_loss_values, 'blue', label='Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    plt.title('Loss for training and validation sets')

    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, acc_values, 'r', label='Training acc')
    ax2.plot(epochs, val_acc_values, 'blue', label='Validation acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.title('Accuracy for training and validation sets')
    plt.show()
