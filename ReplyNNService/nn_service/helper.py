from dynaconf import settings
import requests
import torch
import numpy as np
import regex as re

from nn_service.EmailPartsDetectorLinearModel import EmailPartDetectorLinearModel
from nn_service.reply_extractor.quotations import extract_from_plain, RE_ORIGINAL_MESSAGE
from nn_service.signature_remover import extract_signature

dim = 1024  # LASER's vector size
model = None
class_to_numb = None,
numb_to_class = None


# retrieve vector for the sentence - LASER
def retrieve_embeddings(embed_service, sentences, lang):
    urlstr = embed_service.strip('/') + '/embed'
    urlstr = urlstr + '/' + lang
    if type(sentences) == str:
        sentences = [sentences]
    input_text = {'sentences':sentences}
    response = requests.post(urlstr, json=input_text).json()
    base_embs = response['embedding']
    ''''# convert to numpy array
    base_embs = np.array(base_embs, dtype='float32')
    nbex = base_embs.shape[0] // dim  # number of sentences
    # to reshape to correct size     #embedding = embedding.reshape(nbex,dim)
    base_embs.resize(nbex, dim)'''
    return base_embs


def classify(content, lang):
    """classify  a sentence to find corresponding class

    """
    model, class_to_numb, numb_to_class = load_model()
    top_k = settings['TOP_K']
    try:
        # do the comparison and return index
        embed_service = settings['EMBEDDING_SERVICE']
        embedding_laser = retrieve_embeddings(embed_service, content, lang)
        # tensor
        test_tensors = torch.as_tensor(embedding_laser)
        # print('test_tensors shape', test_tensors.shape)
        # get predictions
        test_preds = model(test_tensors)
        # probabilities
        probabilities = torch.nn.functional.softmax(test_preds, dim=0)  # F.softmin(test_preds, dim=0)
        # get top k predictions
        test_preds = test_preds.detach().numpy()
        probabilities = probabilities.detach().numpy()
        classes = np.argsort(test_preds)[::-1][:top_k]
        classes = classes.tolist()  # convert to standard list for returning
        '''ret_classes = []
        ret_probs = []
        for cls, prbbs in zip(classes, probabilities):
            clas = [numb_to_class[i] for i in cls]  # # converting
            ret_classes.append(clas)
            prbbs = sorted(prbbs, reverse=True)[:top_k]
            ret_probs.append(prbbs)'''
        probabilities = sorted(probabilities.tolist(), reverse=True)[:top_k]

        classes = [numb_to_class[i] for i in classes]  # # converting
        # print('class_names ', classes, 'probabilities ', probabilities)

    except Exception as e:
        # print(e)
        print('Error', e)
        # response.headers.add('Access-Control-Allow-Origin', 'http://localhost')

    return classes, probabilities  if len(classes) > 0 else (['NOT WORKING CLASS 1'], [1.0])


def load_model():
    filepath = settings['MODEL_PATH']
    global model, class_to_numb, numb_to_class
    if model is None:
        model, class_to_numb, numb_to_class = EmailPartDetectorLinearModel.load(filepath)
        print('loading model from ', filepath)
    return model, class_to_numb, numb_to_class


def extract_parts(message, use_nn=False):
    # message = message.strip()  # get rid of extra spaces
    last_message = extract_from_plain(message, use_nn=use_nn)  # .strip()  # use_nn=True
    # print('last_message', last_message)
    prev_thread_part_email = None
    if last_message != message:  # has a part (not single message)
        prev_thread_part_email = message.replace(last_message, '')  # the previous message is the message minus the
        # last message

        # remove last ---- original message ---- line
        if len(last_message.splitlines()) > 0 and re.search(RE_ORIGINAL_MESSAGE, last_message.splitlines()[-1]):
            last_message = '\n'.join(last_message.splitlines()[:-1])
        # remove signature if exists
    last_message_content, signature = extract_signature.extract_signature(last_message, use_nn=use_nn)
    print('signature', signature)

    return last_message_content, signature, prev_thread_part_email
