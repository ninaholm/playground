
#import tensorflow

from sklearn.naive_bayes import GaussianNB
import random
import pickle




def create_trainingdata():
    
    twss = []
    tfln = []
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/TFLNlist.dat", 'rb') as f:
            tfln = pickle.load(f)
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/TWSSlist.dat", 'rb') as f:
            twss = pickle.load(f)
    
    
    
    final_twss = [preprocess_sentence(sent) for sent in twss]
    final_tfln = [preprocess_sentence(sent) for sent in tfln]
    
    x = [[sentence, 1] for sentence in final_twss]
    y = [[sentence, 0] for sentence in final_tfln]
    
    final_dataset = x + y
    random.shuffle(final_dataset)
    
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/PreprocessedData.dat", 'wb') as f:
            pickle.dump(final_dataset, f)
    

def preprocess_sentence(sentence):
    from nltk.corpus import stopwords
    from nltk import word_tokenize
    import string
    
    tokenized_text = word_tokenize(sentence.lower())
    cleaned_text = [word for word in tokenized_text if word not in stopwords.words('english') and word not in string.punctuation]
    
    return(cleaned_text)


def get_baselineVocab(dataset):
    
    vocab = [word for sublist in dataset for word in sublist]
    baseline_vocab = { word : 0 for word in vocab }
    
    for word in vocab:
        baseline_vocab[word] = baseline_vocab[word] + 1
    
    baseline_vocab = { word : 0 for word in baseline_vocab if baseline_vocab[word] > 1}
    print("Total number of unique words in vocabulary: " + str(len(baseline_vocab)))

    with open("C:/Users/NHJ/Desktop/Playground/app/modules/GNB_Vocabulary.dat", 'wb') as f:
        pickle.dump(baseline_vocab, f)
    
    return(baseline_vocab)



def sentence2Features(sentence, baseline_vocab):
    
    for word in sentence:
        if word in baseline_vocab:
            baseline_vocab[word] = 1

    features = list(baseline_vocab.values())
    
    for word in sentence:
        if word in baseline_vocab:
            baseline_vocab[word] = 0

    return(features)



def run_training():
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/PreprocessedData.dat", 'rb') as f:
        final_dataset = pickle.load(f)

    # Downsample TFLN data
    tmp = [sent for sent in final_dataset if sent[1] == 1]
    rand = random.sample([sent for sent in final_dataset if sent[1] == 0], 4200)
    final_dataset = tmp + rand
    random.shuffle(final_dataset)
    
    baseline_vocab = get_baselineVocab([sent[0] for sent in final_dataset])
    
    # Extract x and y data    
    x = [sentence2Features(sent[0], baseline_vocab) for sent in final_dataset]
    y = [sent[1] for sent in final_dataset]
    
    
    # Split into train and test
    cutoff = int(len(final_dataset) * 0.75)
    
    x_train = x[0:cutoff]
    y_train = y[0:cutoff]
    
    x_test = x[cutoff:len(final_dataset)]
    y_test = y[cutoff:len(final_dataset)]
    
    trained_gnb = train_GnbModel(x_train, y_train)
    evaluate_GNB(trained_gnb, x_test, y_test)
    


def classify_text(input_string):
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/Trained_GNB.dat", 'rb') as f:
        trained_gnb = pickle.load(f)
    with open("C:/Users/NHJ/Desktop/Playground/app/modules/GNB_Vocabulary.dat", 'rb') as f:
        baseline_vocab = pickle.load(f)
        
    x = [sentence2Features(preprocess_sentence(input_string), baseline_vocab)]
    result = trained_gnb.predict(x)
    
    return(result[0])
    
    


def train_GnbModel(x_train, y_train):
    gnb = GaussianNB()
    trained_gnb = gnb.fit(x_train, y_train)
    return(trained_gnb)

    with open("C:/Users/NHJ/Desktop/Playground/app/modules/Trained_GNB.dat", 'wb') as f:
            pickle.dump(trained_gnb, f)




def evaluate_GNB(trained_gnb, x_test, y_test):
    from sklearn.metrics import confusion_matrix
    from collections import Counter
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt

    y_pred = trained_gnb.predict(x_test)
    
    confusion_matrix(y_test, y_pred, labels=[0,1])
    
    n_correct = (y_test == y_pred).sum()
    n_all = len(y_test)
    
    print("Actually class distribution: " + str(Counter(y_test)))
    print("Predicted class distribution: " + str(Counter(y_pred)))
    
    print("Number of correctly labeled points out of a total %s points : %d" % (n_all, n_correct))
    
    conf = confusion_matrix(y_test, y_pred, labels=[0,1])
    conf
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    plt.figure()
    plot_confusion_matrix(conf, classes=["0", "1"], title="fisk")
    plt.show()



