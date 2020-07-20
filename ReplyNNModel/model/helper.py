import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


dim = 1024  # LASER's vector size


# count the number of correct predictions (tensor)
def get_correct_predictions(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# plot the confusion matrix for a test
def plot_confusion_matrix(y_true, y_pred, classes, fname_parameters=None, normalize=False, title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # use datetime for naming
    now = datetime.datetime.now()
    filename = './results/conf_matrix_{}_{}_{}.png'.format(
            fname_parameters if fname_parameters is not None else '',
            now.strftime("%Y-%m-%d"),
            'norm' if normalize else 'withoutnorm')
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename)
    return ax


# saving experiment results to a file
def save_result_to_file(file_dir, layer, dropout, epochs, lr, test_correct, test_total, result):
    # write to file
    now = datetime.datetime.now()

    filename = './results/results_class_{}.txt'.format(now.strftime("%Y-%m-%d"))

    if os.path.exists(os.path.join(file_dir, filename)):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    results_file = open(os.path.join(file_dir, filename), append_write)
    if append_write == 'w':  # column names
        results_file.write("layer\tdropout\tepochs\rlearningrate\tcorrect\ttotal\tresult%\n")
    # write results
    results_file.write(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(layer, dropout, epochs, lr, test_correct, test_total, result)
    )
    results_file.close()
    return


# retrieve vector for the sentence - LASER
def retrieve_embeddings(embed_service, sentences, lang):
    urlstr = embed_service.strip('/') + '/embed'
    urlstr = urlstr + '/' + lang
    input_text  = {'sentences':sentences}
    response = requests.post(urlstr, json=input_text).json()
    base_embs = response['embedding']
    return base_embs


# Pass the list with lines containing class and text line
# Returns 2 lists - one list with all classes and one list with all sentences
def get_text_and_classes_from_list(q_list):
    current_quests = [(lambda x: x.strip().split(';', 1))(x) for x in q_list]
    print("length list: ", len(current_quests))
    #logging.warning("length list: ", len(current_quests))
    current_quests = [quest for quest in current_quests if len(quest) == 2]
    classes = [quest[0] for quest in current_quests ]
    questions = [quest[1] for quest in current_quests ]
    print("classes length : ", len(classes), "questions length : ", len(questions))
    return classes, questions


# Load classes of lines in email and the lines themselves
# The line has structure "classID;sentence (or line)"
def load_sentence_and_classes_from_files(file_dir, langs):
    # files are like ecommerce_en.csv
    # os.path.splitext(f)[0][-2:] - gets the lang part
    # os.path.splitext(f)[1][1:] == 'csv' - check if csv
    onlyfiles = [f for f in os.listdir(file_dir) if os.path.isfile(
            os.path.join(file_dir, f)) and os.path.splitext(f)[0][-2:] in langs
                and os.path.splitext(f)[1][1:] == 'csv']
    logging.warning("# of files loaded {}".format(len(onlyfiles)))
    # A dictionary containing langID and line of text in that lang
    lang_questions = {}
    # A dictionary containing langID and classes in that lang
    lang_classes = {}
    for file in onlyfiles:
        logging.warning("File to process {}".format(file))
        current_lang = os.path.splitext(os.path.join(file_dir, file))[0][-2:]
        with open(os.path.join(file_dir, file), encoding='utf-8',
                  errors='surrogateescape') as f:
            texts = f.read().splitlines()  # all lines in the file
            f.close()
            logging.warning(' -   {:s}: {:d} lines'.format(file, len(texts)))
            curr_classes, curr_questions = get_text_and_classes_from_list(texts)
            if lang_questions.get(current_lang) is None:
                # questions by this lang ID do not exist
                lang_questions[current_lang] = curr_questions
            else:
                # merge with existing questions
                logging.debug('before: ', len(lang_questions[current_lang]))
                lang_questions[current_lang] = lang_questions[current_lang] + curr_questions
                logging.debug('after: ', len(lang_questions[current_lang]))
            if lang_classes.get(current_lang) is None:
                # classes by this lang ID do not exist
                lang_classes[current_lang] = curr_classes
            else:
                # merge with existing classes
                lang_classes[current_lang] = lang_classes[current_lang] + curr_classes

    return lang_questions, lang_classes


# retrieve embeddings and corresponding classes of these embeddings
def get_embedded_data_and_classes(embed_service, thedir, langs=['sv'], class_to_numb=None, numb_to_class=None):
    orig_questions, orig_classes = load_sentence_and_classes_from_files(thedir, langs)
    classes = None
    embeddings = None
    if type(langs)==str:
        langs=[langs]
    for lang in langs:
        classes_lang = orig_classes[lang]  # map(int, orig_classes.get(lang)))
        if classes is None:
            classes = classes_lang
        else:
            classes = classes + classes_lang
        # get embeddings
        embedding_laser = retrieve_embeddings(embed_service, orig_questions.get(lang), lang)
        if embeddings is None:
            embeddings = embedding_laser
        else:
            embeddings = embeddings + embedding_laser

    # convert classes to continuous numbers and return those
    if class_to_numb is None:  # new classes - create new dictionaries
        class_to_numb, numb_to_class = create_classes_to_numbers_convertion(classes)
    else:  # some classes exist - add to existing dictionaries
        class_to_numb, numb_to_class = merge_new_classes(classes, class_to_numb, numb_to_class)
    # convert list to numpy array for further conversion to tensor
    embeddings = np.array(embeddings, dtype='float32')

    nbex = embeddings.shape[0] // dim  # number of sentences
    # to reshape to correct size     #embedding = embedding.reshape(nbex,dim)
    embeddings.resize(nbex, dim)
    numbers = [class_to_numb[i] for i in classes]
    return embeddings, numbers, class_to_numb, numb_to_class  #  classes, class_to_numb, numb_to_class


# Converts classes to sequential numbers - the model needs numbers as classes
# Returns two dictionaries - classes to numbers and numbers to classes
def create_classes_to_numbers_convertion(list_classes):
    possible_classes = list(sorted(set(list_classes)))  # make a sorted list of all possible classes
    # print('possibleClasses \t', possibleClasses)
    class_to_numb = dict()
    numb_to_class = dict()
    for i in range(len(possible_classes)):
        class_to_numb[possible_classes[i]] = i
        numb_to_class[i] = possible_classes[i]
    return class_to_numb, numb_to_class


def merge_new_classes(list_classes, class_to_numb, numb_to_class):
    possible_classes = list(sorted(set(list_classes)))  # make a sorted list of all possible classes
    # print('possibleClasses \t', possibleClasses)
    for i in range(len(possible_classes)):
        # if the new class not in the dictionary - add it
        if possible_classes[i] not in class_to_numb:
            class_to_numb[possible_classes[i]] = i
            numb_to_class[i] = possible_classes[i]
    return class_to_numb, numb_to_class
