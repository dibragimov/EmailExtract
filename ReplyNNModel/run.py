import argparse
import copy
import datetime
from itertools import product
import logging
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.optim as torchoptimizer
import torch.nn.functional as func

import model.EmailPartsDetectorLinearModel as EmailPartsDetectorLinearModel
import model.helper as helper


logging.basicConfig(filename='./results/app.log', filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


# train PyTorch NN model
def train_model(lin_lrs, embeddings, classes, dropout=0.0,
                numb_of_epochs=25, lr=0.01, gpu=-1):

    logging.debug("Possible Classes set size: {}".format(len(list(set(classes)))))
    # train model
    our_model = EmailPartsDetectorLinearModel.EmailPartDetectorLinearModel(
        num_lin_lrs=lin_lrs, out_class=len(list(set(classes))), gpu=gpu, dropout=dropout)
    optimizer = torchoptimizer.Adam(our_model.parameters(), lr=lr)

    # TensorBoard initialization
    # comment = f'number of epochs={numbOfEpochs}, layers={lin_lyrs}, dropout={dropout}, learning_rate={lr}'
    # tb = SummaryWriter(comment=comment)
    # logging data
    logging.warning("Embedding length: {}, classes length {}".format(len(embeddings), len(classes)))
    # divide it to train and validation sets
    train_emb, val_emb, train_clss, val_clss = train_test_split(embeddings, classes,
                                                                test_size=0.1, random_state=12)
    # preparing for validation
    best_loss = 0.0
    best_model = None
    # prepare the dataset
    train_tensors = torch.as_tensor(train_emb)
    # make tansor from labels
    label_tensors = torch.from_numpy(np.array(train_clss, dtype='int64'))
    print('shapes', train_tensors.shape, len(train_clss))
    # for training in batches - make a dataset from 2 tensors (embeddings and label)
    united_dataset = torch.utils.data.TensorDataset(train_tensors, label_tensors)
    # load in batches of 54 (27 classes * 2)
    # do the shuffling for each epoch to better train the model
    train_loader = torch.utils.data.DataLoader(united_dataset,
                                               batch_size=5 * len(list(set(classes))),
                                               shuffle=True)
    if gpu >= 0:  # move everuthing to GPU
        device = torch.device("cuda")
        our_model.to(device)
        optimizer.to(device)

    # run training
    for epoch in range(numb_of_epochs):
        total_loss = 0
        total_correct = 0

        # # training for one epoch
        # criterion = nn.CrossEntropyLoss() #### another way of calculating loss
        for batch in train_loader:
            sntncs, lbls = batch
            if gpu >= 0:  # move everuthing to GPU
                device = torch.device("cuda")
                sntncs = sntncs.to(device)
                lbls = lbls.to(device)

            preds = our_model(sntncs)
            loss = func.cross_entropy(preds, lbls)
            # zero all previous values and do the learnimng
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate loss and correct preds during the training
            total_loss += loss.item()
            total_correct += helper.get_correct_predictions(preds, lbls)

        '''#### TensorBoard 
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', (total_loss/len(shuffledLbls)), epoch)

        for name, weight in ourModel.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)'''

        logging.debug("Epoch {}, total_correct {}, total_loss {}".format(
            epoch, total_correct, total_loss))

        # check it against validation loss
        # save the best model with validation loss
        our_model.eval()
        valid_tensors = torch.as_tensor(val_emb)
        valid_labels = torch.from_numpy(np.array(val_clss, dtype='int64'))
        if gpu >= 0:
            device = torch.device("cuda")
            valid_tensors = train_tensors.to(device)
            valid_labels = label_tensors.to(device)
        valid_preds = our_model(valid_tensors)
        valid_loss = func.cross_entropy(valid_preds, valid_labels)
        # assign best loss value != 0 for the first time
        if best_loss == 0:  # for the first time
            best_loss = valid_loss + 1
        # check the best model
        if valid_loss < best_loss:
            logging.warning(
                'The best model according to validation. Loss: {}, epoch: {}, Valid Loss: {}'.format(
                    total_loss, epoch, valid_loss))
            best_loss = valid_loss
            best_model = copy.deepcopy(our_model)
        our_model.train(True)
        # end of validation

    return our_model, best_model


# test the model
def test_model(our_model, test_embedding, testclasses, gpu=-1):
    # make tensor from labels/classes
    test_lbls = torch.from_numpy(np.array(testclasses, dtype='int64'))
    test_nbex_len = len(testclasses)  # total number of questions
    test_tensors = torch.as_tensor(test_embedding)
    if gpu >= 0:
        device = torch.device("cuda")
        test_tensors = test_tensors.to(device)
        test_lbls = test_lbls.to(device)
        our_model = our_model.to(device)
    # get predictions
    our_model.eval()  # otherwise it will go into learning mode
    test_preds = our_model(test_tensors)
    # get correct predictions
    test_total_correct = helper.get_correct_predictions(test_preds, test_lbls)
    logging.debug("testpreds {}\ntest_lbls {}".format(
        test_preds.argmax(dim=1), test_lbls))
    logging.debug(
        "TEST: total {}, total_correct {}, accuracy {}".format(
            test_total_correct, test_nbex_len,
            (test_total_correct / test_nbex_len)
        )
    )
    return test_total_correct, test_nbex_len, (test_total_correct / test_nbex_len), \
           test_lbls.cpu().detach().numpy(), (test_preds.argmax(dim=1)).cpu().detach().numpy()


def run_training():
    is_cuda = torch.cuda.is_available()
    logging.debug("Is CUDA available: {}".format(is_cuda))
    print(("Is CUDA available: {}".format(is_cuda)))
    for lang in ['sv']:
        logging.debug("-------------------DOING IT FOR: {} ------------------".format(lang))
        print("-------------------DOING IT FOR: {} ------------------".format(lang))
        embeddings_all, classes_all, class_to_numb, numb_to_class = helper.get_embedded_data_and_classes(
            embed_service, traindir, langs=lang)
        test_embedding, testclasses, class_to_numb, numb_to_class = helper.get_embedded_data_and_classes(
            embed_service, testdir, langs=lang, class_to_numb=class_to_numb, numb_to_class=numb_to_class)
        logging.debug("Embedding is done")
        #        best_result = 0
        #        best_model = None

        layers = [[700], [500], [300],  # [351], [700], [540],
                  [700, 300], [500, 100],  # [648, 324], [540, 64],
                  # [540, 256, 64], [784, 356, 70], [720, 326, 54], [783, 405, 135],
                  # [720, 500, 280, 81]
                  ]
        dropouts = [0.0, 0.1, 0.2]  # , 0.3]
        epochs = [40, 60, 80]  # 70,50,30,
        lrates = [0.01, 0.001]  # 0.1,
        # build parameters:
        parameters = dict(
            layers=layers,
            dropouts=dropouts,
            epochs=epochs,
            lrates=lrates
        )
        param_values = [v for v in parameters.values()]

        print("One model for all classes")
        best_result = 0
        best_model = None

        testclasses_new, test_embedding_new = testclasses, test_embedding
        # do the experiments for all the parameters
        for layer, drp, ep, lr in product(*param_values):
            the_model, best_val_model = train_model(
                layer, embeddings_all, classes_all, dropout=drp,
                lr=lr, numb_of_epochs=ep, gpu=-1)  # gpu=0 if is_cuda else -1) #
            test_correct, test_total, result, y_true, y_pred = test_model(
                best_val_model if best_val_model is not None else the_model, test_embedding, testclasses,
                gpu=-1)
            if best_val_model is None:
                logging.warning('Validation model is None----------------')
            helper.save_result_to_file(
                '.', layer, drp, ep, lr, test_correct, test_total, result)
            # get the best model
            if result >= best_result:
                logging.debug(
                    '''Selected best model with accuracy {}. 
                    Parameters: layers {}, dropout {}, num_of_epochs {}, learn_rate {}'''.format(
                        result, layer, drp, ep, lr))
                best_result = result
                # make theModel the best validated model. Then save it
                the_model = best_val_model if best_val_model is not None else the_model
                best_model = copy.deepcopy(the_model)
                # if the file exists - delete it. then save the state of the model
                modelfile = './bestmodel/bestmodel_{}_{}.pth'.format(
                    '_'.join(lang),
                    datetime.datetime.now().strftime("%Y-%m-%d"))
                if os.path.exists(modelfile):
                    os.remove(modelfile)
                # when saving - save everything to be able to reload the model
                torch.save({
                    'state_dict': best_model.state_dict(),
                    'num_lin_lrs': layer,
                    'out_class': len(list(set(classes_all))),
                    'dropout': drp,
                    'class_to_numb': class_to_numb,
                    'numb_to_class': numb_to_class
                }, modelfile)
                # now save the confusion matrix
                class_names = np.asarray([i for i in numb_to_class.values()], dtype=np.str)
                # Plot non-normalized confusion matrix
                helper.plot_confusion_matrix(
                    y_true, y_pred, classes=class_names,
                    title='Confusion matrix, without normalization',
                    fname_parameters='_'.join(
                        [str(layer), str(drp), str(ep), str(lr)]
                    )
                )
                # Print the precision and recall, among other metrics
                print(classification_report(y_true, y_pred, digits=3))
                logging.debug(classification_report(y_true, y_pred, digits=3))

        '''# save the whole best model to file
        fullmodelfile = './bestmodel/bestmodel_full_{}_{}.pth'.format(
            '_'.join(lang),
            datetime.datetime.now().strftime("%Y-%m-%d"))
        torch.save(best_model, fullmodelfile)'''


# main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch: EComm Neural network')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Directory for training files')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory for testing files')
    parser.add_argument('--embed-service', type=str, required=True,
                        help='URL of the embedding service')

    args, unknown = parser.parse_known_args()
    embed_service = args.embed_service  # 'http://localhost:7005'
    if args.train_dir:
        traindir = args.train_dir
    if args.test_dir:
        testdir = args.test_dir

    # time.sleep(5 * 1)  # 5 seconds for embedding service to start

    run_training()
