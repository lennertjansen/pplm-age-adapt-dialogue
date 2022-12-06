import argparse
import pdb
from datetime import datetime
import time
from pdb import set_trace
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim

from classifiers import TextClassificationLSTM, TextClassificationBERT, FrozenBERT

from dataset import get_datasets, padded_collate, PadSequence
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter # for logging

from copy import deepcopy
import shutil
from pathlib import Path
import os, glob

import pandas as pd

# for detailed evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
from utils import make_confusion_matrix


# Global variables
# figure saving destiation and dimensions
FIGDIR = 'figures/'
FIGSIZE = (15, 8)


def train_one_epoch(model,
                    model_type,
                    data_loader,
                    criterion,
                    optimizer,
                    device,
                    start_iteration,
                    clip_grad,
                    max_norm,
                    log_interval,
                    losses,
                    accs,
                    writer,
                    disable_bars,
                    epoch):

    # set model to train mode
    model.train()

    # print("\n Starting to train next epoch ... \n")
    batch_index = 0
    for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader, start=start_iteration):

        # move everything to device
        batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                    batch_lengths.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        if model_type == 'lstm':
            # forward pass through model
            log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

            # Evaluate loss, gradients, and update network parameters
            loss = criterion(log_probs, batch_labels)
        elif model_type == 'bert':
            output = model(batch_inputs, batch_labels)
            loss, text_fea = output


        if writer:
            # Add current batch's loss to tensorboard logger
            writer.add_scalar("Loss/train", loss, iteration)

        # Reset gradients for next iteration
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_norm)

        optimizer.step()

        if model_type == 'lstm':
            predictions = torch.argmax(log_probs, dim=1, keepdim=True)
            correct = predictions.eq(batch_labels.view_as(predictions)).sum().item()
            accuracy = correct / log_probs.size(0)
        elif model_type == 'bert':
            # predictions = torch.argmax(text_fea, 1).tolist()
            predictions = torch.argmax(text_fea, 1)
            correct = predictions.eq(batch_labels.view_as(predictions)).sum().item()
            accuracy = correct / batch_labels.size(0)


        # writer.add_scalar("Loss/train", loss, iteration)

        # print(loss.item())
        #
        # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
        #                    Examples/Sec = {:.2f}, "
        #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
        #     datetime.now().strftime("%Y-%m-%d %H:%M"), step,
        #     config.train_steps, config.batch_size, examples_per_second,
        #     accuracy, loss
        # ))

        losses.append(loss.item())
        accs.append(accuracy)

        if writer:
            # Add current batch's accuracy to tensorboard logger
            writer.add_scalar('Accuracy/train', accuracy, iteration)


        if iteration % log_interval == 0:
            # TODO: See this tutorials prettier logging -- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
            # print(
            #     "\n [{}] Iteration {} | Batch size = {} |"
            #     "Average loss = {:.6f} | Accuracy = {:.4f}".format(
            #         datetime.now().strftime("%Y-%m-%d %H:%M"), iteration,
            #         batch_labels.size(0), loss.item(), accuracy
            #     )
            # )

            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| loss {:8.5f} '
                  '| accuracy {:8.5f} '.format(epoch + 1, batch_index, len(data_loader), loss.item(), accuracy))

        batch_index+=1

    return iteration, losses, accs



def train(seed,
          data,
          model_type,
          mode,
          device,
          batch_size,
          embedding_dim,
          hidden_dim,
          num_layers,
          bidirectional,
          dropout,
          batch_first,
          epochs,
          lr,
          clip_grad,
          max_norm,
          early_stopping_patience,
          train_frac,
          val_frac,
          test_frac,
          subset_size,
          log_interval,
          no_tb,
          w_loss,
          w_sampling,
          writer=None,
          train_dataset=None,
          val_dataset=None,
          test_dataset=None):


    if mode =='train' or mode == 'test':
        # set seed for reproducibility on cpu or gpu based on availability
        torch.manual_seed(seed) if device == 'cpu' else torch.cuda.manual_seed(seed)

        # data_path = '
        # /bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'
        # data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'
        if data == 'bnc':
            data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
        elif data == 'bnc_rb':
            data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
        else:
            data_path = 'data/blogs_kaggle/blogtext.csv'

        # set starting time of full training pipeline
        start_time = datetime.now()

        # set device
        device = torch.device(device)
        print(f"Device: {device}")

        print("Starting data preprocessing ... ")
        data_prep_start = datetime.now()

        # Load data and create dataset instances
        train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                                file_path=data_path,
                                                                train_frac=train_frac,
                                                                val_frac=val_frac,
                                                                test_frac=test_frac,
                                                                seed=seed,
                                                                data=data,
                                                                model_type=model_type)

        # Train, val, and test splits
        # train_size = int(train_frac * len(dataset))
        # test_size = len(dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


        # # get vocab size number of classes
        # vocab_size = train_dataset.vocab_size
        # num_classes = train_dataset.num_classes

        # create dataloaders with pre-specified batch size
        # data_loader = DataLoader(dataset=dataset,
        #                          batch_size=batch_size,
        #                          shuffle=True,
        #                          collate_fn=PadSequence())


        print('-' * 91)
        print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

        print(31*'-' + ' DATASET STATS AND BASELINES ' + '-'*31)
        print(f'Number of classes: {train_dataset.num_classes}')
        print(f'Vocabulary size: {train_dataset.vocab_size}')
        print(f'Training set size: {train_dataset.__len__()}')
        print(f'Validation set size: {val_dataset.__len__()}')
        print(f'Test set size: {test_dataset.__len__()}')
        print(91 * '-')
        print('Baselines')
        print('Train')
        print(train_dataset.df['age_cat'].value_counts(normalize=True))
        print('Validation')
        print(val_dataset.df['age_cat'].value_counts(normalize=True))
        print('Test')
        print(test_dataset.df['age_cat'].value_counts(normalize=True))
        print('-' * 91)


    if w_sampling:
        # Apply weighted sampling.

        # Inspired by: https://towardsdatascience.com/address-class-imbalance-easily-with-pytorch-e2d4fa208627

        # TODO: isn't this a bit redundant? Doesn't torch.tensor(train_dataset.df['age_cat'], dtype=torch.long) do the same?
        all_label_ids = torch.tensor([label for label in train_dataset.df['age_cat']], dtype=torch.long)

        # Class weighting
        labels_unique, counts = np.unique(train_dataset.df['age_cat'], return_counts=True)
        print(f'Unique labels: {labels_unique}')

        class_weights = [sum(counts) / c for c in counts] # [#{class_0}, {#class_1}, etc.]

        # Assign weights to each input sample
        sampler_weights = [class_weights[label] for label in train_dataset.df['age_cat']]
        sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(train_dataset.df['age_cat']), replacement=True)

        # Note that sampler option is mutually exclusive with shuffle. So shuffle not needed here.
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=PadSequence(),
                                  sampler=sampler)
    else:

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=PadSequence())

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=PadSequence())

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=PadSequence())

    if mode == 'train' or mode == 'val' or mode == 'tvt':

        if model_type == 'lstm':
            # initialize model
            print("Initializing model ...")
            model = TextClassificationLSTM(batch_size = batch_size,
                                           vocab_size = train_dataset.vocab_size,
                                           embedding_dim = embedding_dim,
                                           hidden_dim = hidden_dim,
                                           num_classes = train_dataset.num_classes,
                                           num_layers = num_layers,
                                           bidirectional = bidirectional,
                                           dropout = dropout,
                                           device = device,
                                           batch_first = batch_first)

        elif model_type == 'bert':
            model = TextClassificationBERT(num_classes = train_dataset.num_classes)

            # freeze all the BERT-parameters
            # for param in model.encoder.bert.parameters():
            #     param.requires_grad = False

    elif mode == 'test':
        if model_type == 'lstm':
            model, _, _, _, _ = load_saved_model(model_class=TextClassificationLSTM, optimizer_class=optim.Adam, lr=lr,
                                                 device=device, batch_size=batch_size, vocab_size=train_dataset.vocab_size,
                                                 embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                                 num_classes=train_dataset.num_classes, num_layers=num_layers,
                                                 bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)
        elif model_type == 'bert':
            model = TextClassificationBERT()



    # model to device
    model.to(device)


    # Print model architecture and trainable parameters
    print('-' * 91)
    print("MODEL ARCHITECTURE:")
    print(model)
    print('-' * 91)

    if w_loss:

        # Apply frequency-based weighted loss for highly imbalanced data
        n_samples = [train_dataset.df['age_cat'].value_counts()[label] for label in range(train_dataset.num_classes)]

        # Weight option 1
        weights = [1 - (x / sum(n_samples)) for x in n_samples]
        weights = torch.FloatTensor(weights).to(device)


        # OR 2) have the weights sum up to 1??
        # weights = torch.tensor(n_samples, dtype=torch.float32).to(device)
        # weights = weights / weights.sum()
        # weights = 1.0 / weights
        # weights = weights / weights.sum()


        criterion = torch.nn.CrossEntropyLoss(weight=weights)  # combines LogSoftmax and NLL
    else:
        criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    if mode == 'train' or mode == 'val' or mode == 'tvt':

        # count trainable parameters
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'The model has {trainable_params} trainable parameters.')

        # set up optimizer and loss criterion
        if model_type == 'lstm':
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
        elif model_type == 'bert':
            optimizer = optim.Adam(model.parameters(), lr=2e-5) #TODO: CHANGE THIS BACK!!!!!!!

        # initialize iterations at zero
        iterations = 0

        # values for model selection
        best_val_loss = torch.tensor(np.inf, device=device)
        best_val_accuracy = torch.tensor(-np.inf, device=device)
        best_epoch = None
        best_model = None

        # Initialize patience for early stopping
        patience = 0

        # metrics for losses
        train_losses = []
        train_accs = []

        # disable tqdm progress bars in train and train_one_epoch if in validation mode
        disable_bars = mode == 'val'

        # for epoch in tqdm(range(epochs), disable=disable_bars):
        for epoch in range(epochs):

            epoch_start_time = datetime.now()
            # epoch_start_time = time.time()
            try:
                # set model to training mode. NB: in the actual training loop later on, this
                # statement goes at the beginning of each epoch.
                model.train()
                iterations, train_losses, train_accs = train_one_epoch(model=model, model_type=model_type,
                                                                       data_loader=train_loader,
                                                                       criterion=criterion,
                                                                       optimizer=optimizer, device=device,
                                                                       start_iteration=iterations,
                                                                       clip_grad=clip_grad, max_norm=max_norm,
                                                                       log_interval=log_interval,
                                                                       losses=train_losses, accs=train_accs, writer=writer,
                                                                       disable_bars=disable_bars,
                                                                       epoch=epoch)

            except KeyboardInterrupt:
                print("Manually stopped current epoch")
                __import__('pdb').set_trace()

            # print("Current epoch training took {}".format(datetime.now() - epoch_start_time))
            val_loss, val_accuracy = evaluate_performance(model=model,
                                                          data_loader=val_loader,
                                                          device=device,
                                                          criterion=criterion,
                                                          writer=writer,
                                                          global_iteration=iterations,
                                                          print_metrics=False,
                                                          data=data,
                                                          model_type=model_type,
                                                          mode='val')
            # TODO: See this tutorials prettier logging -- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
            # print(f"#######################################################################")
            # print(f"Epoch {epoch + 1} finished, validation loss: {val_loss}, val acc: {val_accuracy}")
            # print(f"#######################################################################")

            print('-' * 91)
            print('| end of epoch {:3d} | time: {} | '
                  'valid loss {:8.5f} | valid accuracy {:8.3f} '.format(epoch + 1,
                                                                        (datetime.now() - epoch_start_time),
                                                                        val_loss, val_accuracy))
            print('-' * 91)

            # # update best performance
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_val_accuracy = val_accuracy
            #     best_model = model
            #     best_epoch = epoch + 1

            # update best performance
            if val_accuracy > best_val_accuracy:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_model = deepcopy(model)
                best_optimizer = deepcopy(optimizer)
                best_epoch = epoch + 1
                patience = 0
            else:
                patience +=1
                if patience >= early_stopping_patience:
                    print("EARLY STOPPING")
                    break
        # TODO: See this tutorials prettier logging -- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
        print('-' * 91)
        print(f"Done training and validating. Best model from epoch {best_epoch}:")
        # print(best_model)
        print('-' * 91)

    if mode == 'val':
        return best_val_loss, best_val_accuracy, best_model, best_epoch, best_optimizer
    elif mode == 'train':
        print('Saving best model...')
        cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        save_path = f'models/{data}/{model_type}/{model_type}_{data}_ws_ca_seed_{seed}_{cur_datetime}.pt'
        torch.save(best_model.state_dict(), save_path)
        print("Starting testing...")
        _, _ = evaluate_performance(model=best_model, data_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion,
                                                  set='test',
                                                  data=data,
                                                  plot_cm=True,
                                                  model_type=model_type, mode=mode)
    elif mode == 'test':
        print("Starting testing...")
        _, _ = evaluate_performance(model=model, data_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion,
                                                  set='test',
                                                  data=data,
                                                  plot_cm=True,
                                                  model_type=model_type,
                                                  mode=mode)
    elif mode == 'tvt':
        print('Evaluating model performance...')
        loss, accuracy, f1_scores = evaluate_performance(model=best_model, data_loader=val_loader,
                                                         device=device,
                                                         criterion=criterion,
                                                         set='val',
                                                         data=data,
                                                         plot_cm=False,
                                                         model_type=model_type,
                                                         mode=mode)
        return loss, accuracy, f1_scores, best_model, criterion


    # plot_performance(losses=train_losses, accs=train_accs)





def evaluate_performance(model, data_loader, device, criterion, data, writer=None, global_iteration=0, set='validation',
                         print_metrics=True, plot_cm=False, save_fig=True, show_fig=False, model_type='lstm', mode='val'):
    # For Confucius matrix
    y_pred = []
    y_true = []

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    set_loss = 0
    total_correct = 0

    # start eval timer
    eval_start_time = datetime.now()

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader):

            # move everything to device
            batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                        batch_lengths.to(device)

            if model_type == 'lstm':
                # forward pass through model
                log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

                # compute and sum up batch loss and correct predictions
                set_loss += criterion(log_probs, batch_labels)



                # predictions = torch.argmax(log_probs, dim=1, keepdim=True) # Old
                # batch_pred = [int(item[0]) for item in predictions.tolist()] # Old
                predictions = torch.argmax(log_probs, dim=1) # New
                # pdb.set_trace()
                # predictions = torch.randint(0, 2, predictions.shape)

            elif model_type == 'bert':
                loss, text_fea = model(batch_inputs, batch_labels)
                set_loss += loss

                predictions = torch.argmax(text_fea, 1)


            # batch_pred = [int(item[0]) for item in predictions.tolist()]
            # batch_pred = predictions.tolist()
            # ## OLD
            # if model_type == 'lstm':
            #     y_pred.extend(batch_pred)
            # elif model_type == 'bert':
            #     y_pred.extend(predictions.tolist())

            y_pred.extend(predictions.tolist()) #New
            y_true.extend(batch_labels.tolist())

            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # average losses and accuracy
        set_loss /= len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)
        if print_metrics:
            print('-' * 91)
            print(
                "| " + set + " set "
                "| time {}"
                "| loss: {:.5f} | Accuracy: {}/{} ({:.5f})".format(
                    datetime.now() - eval_start_time, set_loss, total_correct, len(data_loader.dataset), accuracy
                )
            )
            print('-' * 91)

        if writer:
            if set == 'validation':
                writer.add_scalar('Accuracy/val', accuracy, global_iteration)
                writer.add_scalar('Loss/val', set_loss, global_iteration)

        print(91 * '-')
        print(34 * '-' + ' Classification Report ' + 34 * '-')
        labels = [label for label in range(data_loader.dataset.num_classes)]
        print(classification_report(y_true, y_pred, labels=labels, digits=5, zero_division=0))

        print(91 * '-')
        print('| Confusion Matrix |')
        # cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # df_confusion = pd.DataFrame(cm * len(y_true))
        df_confusion = pd.DataFrame(cm)
        print("    Predicted")
        print(df_confusion)
        print("True -->")

        # print(cm * len(y_true))

        if plot_cm:

            if data == 'bnc' or 'bnc_rb':
                tick_labels = ['19_29', '50_plus']
            elif data == 'blog':
                tick_labels = ['13-17', '23-27', '33-47']
            make_confusion_matrix(cf=cm, categories=tick_labels, title=f'Confusion Matrix for {data} on {set} set',
                                  num_labels=labels, y_true=y_true, y_pred=y_pred, figsize=FIGSIZE)

            if save_fig:
                cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                plt.savefig(f"{FIGDIR}{data}/cm_{model_type}_{set}_dt_{cur_datetime}.png",
                            bbox_inches='tight')
            if show_fig:
                plt.show()


        # if plot_cm:
        #     #TODO: implement this function make_confusion_matrix
        #     # link: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
        #     ax = plt.subplot()
        #     sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.3f')
        #
        #     ax.set_title('Confusion Matrix')
        #
        #     ax.set_xlabel('Predicted Labels')
        #     ax.set_ylabel('True Labels')
        #
        #
        #     if data == 'bnc':
        #         tick_labels = ['19_29', '50_plus']
        #         ax.xaxis.set_ticklabels(tick_labels)
        #         ax.yaxis.set_ticklabels(tick_labels)
        #     elif data == 'blog':
        #         tick_labels = ['13-17', '23-27', '33-47']
        #         ax.xaxis.set_ticklabels(tick_labels)
        #         ax.yaxis.set_ticklabels(tick_labels)
        #
        #     plt.show()

        if mode == 'tvt':
            f1_scores = f1_score(y_true, y_pred, average=None)

            return set_loss, accuracy, f1_scores
        else:
            return set_loss, accuracy


def plot_performance(losses, accs, show=False, save=False):
    # saving destiation
    FIGDIR = './figures/'
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 12))
    fig.suptitle(f"LSTM accuracy and loss for default settings.")

    # accs_run, loss_run, steps_run = train(config, seed=seed)
    # accs_runs.append(accs_run)
    # loss_runs.append(loss_run)
    # steps_runs.append(steps_run)
    #
    # accs_means = np.mean(accs_runs, axis=0)
    # accs_stds = np.std(accs_runs, axis=0)
    # ci = 1.96 * accs_stds / np.sqrt(num_runs)
    #
    # loss_means = np.mean(loss_runs, axis=0)
    # loss_stds = np.std(loss_runs, axis=0)
    # ci_loss = 1.96 * loss_stds / np.sqrt(num_runs)

    steps = np.arange(len(losses))

    ax1.plot(steps, accs, label="Average accuracy")
    # ax1.fill_between(steps_runs[0], (accs_means - ci), (accs_means + ci), alpha=0.3)
    ax1.set_title("Accuracy and Loss for character prediction.")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("steps")

    ax2.plot(steps, losses, label="Average CE Loss")
    # ax2.fill_between(steps_runs[0], (loss_means - ci_loss),
    #                  (loss_means + ci_loss), alpha=0.3)
    ax2.set_title("CE Loss for various sequence lengths")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("steps")

    ax1.legend()
    ax2.legend()

    if save:
        plt.savefig(f"{FIGDIR}lstm_blog.png",
                    bbox_inches='tight')
    if show:
        plt.show()


def hp_search(seed,
              data,
              model_type,
              mode,
              device,
              batch_size,
              embedding_dim,
              hidden_dim,
              num_layers,
              bidirectional,
              dropout,
              batch_first,
              epochs,
              lr,
              clip_grad,
              max_norm,
              early_stopping_patience,
              train_frac,
              val_frac,
              test_frac,
              subset_size,
              log_interval,
              no_tb,
              w_loss,
              w_sampling):

    # set seed for reproducibility on cpu or gpu based on availability
    torch.manual_seed(seed) if device == 'cpu' else torch.cuda.manual_seed(seed)

    # set starting time of full training pipeline
    start_time = datetime.now()

    # set device
    device = torch.device(device)
    print(f"Device: {device}")

    # data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'

    if data == 'bnc':
        data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
    elif data == 'bnc_rb':
        data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
    else:
        data_path = 'data/blogs_kaggle/blogtext.csv'

    print("Starting data preprocessing ... ")
    data_prep_start = datetime.now()

    # Load data and create dataset instances
    train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                            file_path=data_path,
                                                            train_frac=train_frac,
                                                            val_frac=val_frac,
                                                            test_frac=test_frac,
                                                            seed=seed,
                                                            data=data)

    print('-' * 91)
    print('BASELINES//VALUE COUNTS')
    print('Train')
    print(train_dataset.df['age_cat'].value_counts(normalize=True))
    print('Validation')
    print(val_dataset.df['age_cat'].value_counts(normalize=True))
    print('-' * 91)

    # Train, val, and test splits
    # train_size = int(train_frac * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # get vocab size number of classes
    # vocab_size = train_dataset.vocab_size
    # num_classes = train_dataset.num_classes

    # create dataloaders with pre-specified batch size
    # data_loader = DataLoader(dataset=dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=PadSequence())

    print('-' * 91)
    print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

    print('######### DATA STATS ###############')
    print(f'Number of classes: {train_dataset.num_classes}')
    print(f'Vocabulary size: {train_dataset.vocab_size}')
    print(f'Training set size: {train_dataset.__len__()}')
    print(f'Validation set size: {val_dataset.__len__()}')
    print(f'Test set size: {test_dataset.__len__()}')
    print('-' * 91)

    # Set hyperparameters for grid search*
    # seeds = [0, 1, 2]
    lrs = [1e-3, 1e-2]
    # lrs = [0.001]
    embedding_dims = [64, 256, 512]
    # embedding_dims = [512]
    hidden_dims = [128, 512, 1024]
    # hidden_dims = [1024]
    nums_layers = [1, 2]
    # nums_layers = [2]
    bidirectionals = [False, True]
    # bidirectionals = [False]
    # weighting = [(True, False), (False, True)] # [(w_loss = True, w_sampling = False), (w_loss = False, w_sampling = True)]


    # set holders for best performance metrics and corresponding hyperparameters
    best_metrics = {'loss' : float("inf"),
                    'acc' : float('-inf')}
    best_hps = {'lr' : None,
                'embedding_dim' : None,
                'hidden_dim' : None,
                'num_layers': None,
                'bidirectional' : None}

    best_model = None # TODO: what's the appropriate type for this?

    #TODO: add tqdm's and print statements to these loops for progress monitoring

    best_file_name = None
    best_epoch = None

    # For keeping track of metrics for all configs
    keys = ['lr', 'emb_dim', 'hid_dim', 'n_layers', 'bd', 'val_acc', 'val_loss']
    df = pd.DataFrame(columns=keys)

    best_model_updates = -1

    for lr_ in lrs:
        for emb_dim in embedding_dims:
            for hid_dim in hidden_dims:
                # skip if hidden size not larger than embedding dim
                if not hid_dim > emb_dim:
                    continue

                for n_layers in nums_layers:
                    for bd in bidirectionals:

                        print('-' * 91)
                        print(f"| Current config: lr: {lr_} | emb: {emb_dim} | hid_dim: {hid_dim} | n_layers: {n_layers} "
                              f"| bd: {bd} | ")
                        print('-' * 91)


                        # Create detailed experiment tag for tensorboard summary writer
                        cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                        file_name = f'lstm_emb_{emb_dim}_hid_{hid_dim}_l_{n_layers}_' \
                                    f'bd_{bd}_drop_{dropout}_bs_{batch_size}_epochs_{epochs}_' \
                                    f'lr_{lr_}_subset_{subset_size}_train_{train_frac}_val_{val_frac}_' \
                                    f'test_{test_frac}_clip_{clip_grad}_maxnorm_{max_norm}' \
                                    f'es_{early_stopping_patience}_seed_{seed}_device_{device}_dt_{cur_datetime}'

                        if not no_tb:
                            # # Create detailed experiment tag for tensorboard summary writer
                            # cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                            log_dir = f'runs/hp_search/{data}/'
                            # file_name = f'lstm_emb_{emb_dim}_hid_{hid_dim}_l_{n_layers}_' \
                            #             f'bd_{bd}_drop_{dropout}_bs_{batch_size}_epochs_{epochs}_' \
                            #             f'lr_{lr_}_subset_{subset_size}_train_{train_frac}_val_{val_frac}_' \
                            #             f'test_{test_frac}_clip_{clip_grad}_maxnorm_{max_norm}' \
                            #             f'es_{early_stopping_patience}_seed_{seed}_device_{device}_dt_{cur_datetime}'

                            # create summary writer instance for logging
                            log_path = log_dir+file_name
                            writer = SummaryWriter(log_path)
                        else:
                            writer = None

                        # train model (in val mode)
                        loss, acc, model, epoch, optimizer = train(mode=mode, data=data, seed=seed, device=device,
                                                                   batch_size=batch_size, embedding_dim=emb_dim,
                                                                   hidden_dim=hid_dim, num_layers=n_layers,
                                                                   bidirectional=bd, dropout=dropout,
                                                                   batch_first=batch_first, epochs=epochs,
                                                                   lr=lr_, clip_grad=clip_grad, max_norm=max_norm,
                                                                   early_stopping_patience=early_stopping_patience,
                                                                   train_frac=train_frac, val_frac=val_frac,
                                                                   test_frac=test_frac, subset_size=subset_size,
                                                                   log_interval=log_interval, writer=writer,
                                                                   train_dataset=train_dataset, val_dataset=val_dataset,
                                                                   test_dataset=test_dataset, no_tb=no_tb, w_loss=w_loss,
                                                                   w_sampling=w_sampling, model_type=model_type)

                        if not no_tb:
                            # close tensorboard summary writer
                            writer.close()


                        # Update metric logging dataframe
                        df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [lr_] + [emb_dim] + [hid_dim] \
                                                                                         + [n_layers] + [bd] + [acc] + \
                                                                                         [loss.item()]

                        # Save metric logging dataframe to csv
                        # cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                        df.to_csv(
                            f'output/{data}_lstm_hp_search_metrics_ws.csv', index=False
                        )

                        # update best ...
                        if acc > best_metrics['acc']:

                            best_model_updates +=1

                            # ... metrics
                            best_metrics['acc'] = acc
                            best_metrics['loss'] = loss

                            best_epoch = epoch

                            # ... hyperparams
                            best_hps['lr'] = lr_
                            best_hps['embedding_dim'] = emb_dim
                            best_hps['hidden_dim'] = hid_dim
                            best_hps['num_layers'] = n_layers
                            best_hps['bidirectional'] = bd

                            # ... model
                            best_model = deepcopy(model)

                            # ... optimizer
                            best_optimizer = deepcopy(optimizer)

                            # filename
                            best_file_name = file_name

                            # Delete previous current best model checkpoint file
                            for filename in glob.glob(f"models/{data}/lstm/cur_best_*"):
                                os.remove(filename)

                                # save current best model checkpoint
                            # Save best model checkpoint
                            model_dir = f'models/{data}/lstm/'
                            Path(model_dir).mkdir(parents=True, exist_ok=True)
                            model_path = model_dir + 'cur_best_' + best_file_name + '.pt'

                            torch.save({
                                'epoch': best_epoch,
                                'model_state_dict': best_model.state_dict(),
                                'optimizer_state_dict': best_optimizer.state_dict(),
                                'loss': best_metrics['loss'],
                                'acc': best_metrics['acc']
                            }, model_path)

                            print("New current best model found.")
                            print(f'Current best hyperparameters: {best_hps}')
                            print(f'Current best model: {best_model}')
                            print(f'Current best metrics: {best_metrics}')

    # # Save metric logging dataframe to csv
    # df.to_csv(
    #     'output/blog_lstm_hp_search_metrics.csv',
    #     index=False
    # )

    # Save best model checkpoint
    model_dir = f'models/{data}/lstm/'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = model_dir + 'best_' + best_file_name + '.pt'

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': best_optimizer.state_dict(),
        'loss': best_metrics['loss'],
        'acc': best_metrics['acc']
    }, model_path)

    print("Finished hyperparameter search.")
    print(f'Best hyperparameters: {best_hps}')
    print(f'Best model: {best_model}')
    print(f'Best metrics: {best_metrics}')

    # Delete equivalent cur_best file
    for filename in glob.glob("models/blog/lstm/cur_best_*"):
        os.remove(filename)

    print(f"Best model updates: {best_model_updates}")



def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_saved_model(model_class, optimizer_class, lr, device, batch_size, vocab_size, embedding_dim, hidden_dim,
                     num_classes, num_layers, bidirectional, dropout, batch_first):

    # blog lstm
    # checkpoint_path = 'models/blog/lstm/best_blog_lstm_emb_128_hid_256_l_2_bd_True_drop_0_bs_64_epochs_5_lr_0.001_' \
    #                   'subset_None_train_0.75_val_0.15_test_0.1_clip_False_maxnorm_10.0es_2_seed_2021_' \
    #                   'device_cuda_dt_13_May_2021_16_25_34.pt'
    #
    # # w_loss
    # checkpoint_path = 'models/bnc/lstm/best_lstm_emb_256_hid_512_l_1_bd_True_drop_0_bs_64_epochs_15_lr_0.0001_' \
    #                   'subset_None_train_0.75_val_0.15_test_0.1_clip_False_maxnorm_10.0es_2_seed_2021_device_cuda_' \
    #                   'dt_23_May_2021_01_00_29.pt'

    # w_sampling
    # checkpoint_path = 'models/bnc/lstm/best_lstm_emb_256_hid_1024_l_1_bd_False_drop_0_bs_64_epochs_15_lr_0.0001_' \
    #                   'subset_None_train_0.75_val_0.15_test_0.1_clip_False_maxnorm_10.0es_2_seed_2021_' \
    #                   'device_cuda_dt_22_May_2021_01_31_46.pt'

    # bnc_rb
    checkpoint_path = 'models/bnc_rb/lstm/best_lstm_emb_512_hid_1024_l_2_bd_False_drop_0_bs_64_epochs_15_lr_0.001_' \
                      'subset_None_train_0.75_val_0.15_test_0.1_clip_False_maxnorm_10.0es_2_seed_2021_' \
                      'device_cuda_dt_23_May_2021_21_58_42.pt'



    # initialize model instance
    model = model_class(batch_size=batch_size, vocab_size=vocab_size, embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout, device=device, batch_first=batch_first)

    # model to device
    model.to(device)

    # initialize optimizer
    optimizer = optimizer_class(params=model.parameters(), lr=lr)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # {
    #     'epoch': best_epoch,
    #     'model_state_dict': best_model.state_dict(),
    #     'optimizer_state_dict': best_optimizer.state_dict(),
    #     'loss': best_metrics['loss'],
    #     'acc': best_metrics['acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']


    return model, optimizer, epoch, loss, acc


def train_test_val(seed,
                  data,
                  model_type,
                  mode,
                  device,
                  batch_size,
                  embedding_dim,
                  hidden_dim,
                  num_layers,
                  bidirectional,
                  dropout,
                  batch_first,
                  epochs,
                  lr,
                  clip_grad,
                  max_norm,
                  early_stopping_patience,
                  train_frac,
                  val_frac,
                  test_frac,
                  subset_size,
                  log_interval,
                  no_tb,
                  w_loss,
                  w_sampling):

    # set starting time of full training pipeline
    start_time = datetime.now()

    # Random seeds for reproducibility and averaging over different initializations
    # seeds = [1, 2, 3, 4, 5]
    seeds = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]

    # For keeping track of metrics for all configs
    if data == 'bnc' or data == 'bnc_rb':
        keys = ['seed', 'val_loss', 'val_acc', 'val_f1_0', 'val_f1_1',
                        'test_loss', 'test_acc', 'test_f1_0', 'test_f1_1']
    elif data == 'blog':
        keys = ['seed', 'val_loss', 'val_acc', 'val_f1_0', 'val_f1_1', 'val_f1_2'
                        'test_loss', 'test_acc', 'test_f1_0', 'test_f1_1', 'test_f1_2']

    df = pd.DataFrame(columns=keys)

    # Create detailed experiment tag for tensorboard summary writer
    cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
    file_name = f'{model_type}_emb_{embedding_dim}_hid_{hidden_dim}_l_{num_layers}_' \
                f'bd_{bidirectional}_drop_{dropout}_bs_{batch_size}_epochs_{epochs}_' \
                f'lr_{lr}_subset_{subset_size}_train_{train_frac}_val_{val_frac}_' \
                f'test_{test_frac}_clip_{clip_grad}_maxnorm_{max_norm}' \
                f'es_{early_stopping_patience}_device_{device}_dt_{cur_datetime}_'

    save_destination = f'output/{data}/' + file_name + 'val_test_metrics.csv'

    for seed_ in seeds:

        # set seed for reproducibility on cpu or gpu based on availability
        torch.manual_seed(seed_) if device == 'cpu' else torch.cuda.manual_seed(seed_)

        # set device
        device = torch.device(device)
        print(f"Device: {device}")

        # data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'

        if data == 'bnc':
            data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
        elif data == 'bnc_rb':
            data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
        elif data == 'blog':
            data_path = 'data/blogs_kaggle/blogtext.csv'

        print("Starting data preprocessing ... ")
        data_prep_start = datetime.now()

        # Load data and create dataset instances
        train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                                file_path=data_path,
                                                                train_frac=train_frac,
                                                                val_frac=val_frac,
                                                                test_frac=test_frac,
                                                                seed=seed_,
                                                                data=data)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=PadSequence())

        print('-' * 91)
        print('BASELINES//VALUE COUNTS')
        print('Train')
        print(train_dataset.df['age_cat'].value_counts(normalize=True))
        print('Validation')
        print(val_dataset.df['age_cat'].value_counts(normalize=True))
        print('Test')
        print(test_dataset.df['age_cat'].value_counts(normalize=True))
        print('-' * 91)

        print('-' * 91)
        print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

        print('######### DATA STATS ###############')
        print(f'Number of classes: {train_dataset.num_classes}')
        print(f'Vocabulary size: {train_dataset.vocab_size}')
        print(f'Training set size: {train_dataset.__len__()}')
        print(f'Validation set size: {val_dataset.__len__()}')
        print(f'Test set size: {test_dataset.__len__()}')
        print('-' * 91)

        writer = None

        val_loss, val_acc, val_f1_scores, model, criterion = train(mode=mode, model_type=model_type, data=data,
                                                                   seed=seed_, device=device,
                                                                   batch_size=batch_size, embedding_dim=embedding_dim,
                                                                   hidden_dim=hidden_dim, num_layers=num_layers,
                                                                   bidirectional=bidirectional, dropout=dropout,
                                                                   batch_first=batch_first, epochs=epochs,
                                                                   lr=lr, clip_grad=clip_grad, max_norm=max_norm,
                                                                   early_stopping_patience=early_stopping_patience,
                                                                   train_frac=train_frac, val_frac=val_frac,
                                                                   test_frac=test_frac, subset_size=subset_size,
                                                                   log_interval=log_interval, writer=writer,
                                                                   train_dataset=train_dataset, val_dataset=val_dataset,
                                                                   test_dataset=test_dataset, no_tb=no_tb, w_loss=w_loss,
                                                                   w_sampling=w_sampling)

        # get test metrics
        test_loss, test_acc, test_f1_scores = evaluate_performance(model=model, data_loader=test_loader, device=device,
                                                                   criterion=criterion, set='test', data=data,
                                                                   plot_cm=True, model_type=model_type, mode=mode)

        #update dataframe

        # if data == 'bnc' or data == 'bnc_rb':
        #     keys = ['seed', 'val_loss', 'val_acc', 'val_f1_0', 'val_f1_1',
        #             'test_loss', 'test_acc', 'test_f1_0', 'test_f1_1']
        # elif data == 'blog':
        #     keys = ['seed', 'val_loss', 'val_acc', 'val_f1_0', 'val_f1_1', 'val_f1_2'
        #             'test_loss', 'test_acc', 'test_f1_0', 'test_f1_1', 'test_f1_2']

        # Update metric logging dataframe
        if data == 'bnc' or data == 'bnc_rb':
            df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [seed_] + [val_loss.item()] + [val_acc] + \
                                                                             [val_f1_scores[0]] + [val_f1_scores[1]] + \
                                                                             [test_loss.item()] + [test_acc] + \
                                                                             [test_f1_scores[0]] + [test_f1_scores[1]]
        elif data == 'blog':
            df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [seed_] + [val_loss.item()] + [val_acc] + \
                                                                             [val_f1_scores[0]] + [val_f1_scores[1]] + \
                                                                             [val_f1_scores[2]] + \
                                                                             [test_loss.item()] + [test_acc] + \
                                                                             [test_f1_scores[0]] + [test_f1_scores[1]] + \
                                                                             [test_f1_scores[2]]

        # Save metric logging dataframe to csv
        # cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        df.to_csv(
            save_destination, index=False
        )



def parse_arguments(args = None):
    parser = argparse.ArgumentParser(description="Train discriminator neural text classifiers.")

    parser.add_argument(
        '--data', type=str, choices=['blog', 'bnc', 'bnc_rb'], default='bnc_rb',
        help='Choose dataset to work with. Either blog corpus, BNC (NB: highly imbalanced. '
             'Activate w_loss or w_sampling), or BNC randomly balanced (bnc_rb).'
    )
    parser.add_argument(
        '--model_type', type=str, choices=['lstm', 'bert', 'unigram'], default='lstm',
        help='Model type/class to use.'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'val', 'test', 'tvt'], default='train',
        help='Set script to training, development/validation, or test mode.'
    )
    parser.add_argument(
        "--seed", type=int, default=2021, help="Seed for reproducibility"
    )
    parser.add_argument(
        '--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
        help="Device to run the model on."
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help="Number of datapoints to simultaneously process."
    )  # TODO: set to reasonable default after batching problem fixed.
    parser.add_argument(
        '--embedding_dim', type=int, default=16, help="Dimensionality of embedding."
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=32, help="Size in LSTM hidden layer."
    )
    parser.add_argument(
        '--num_layers', type=int, default=1, help='Number of hidden layers in LSTM'
    )
    parser.add_argument(
        '--bidirectional', action='store_true',
        help='Create a bidirectional LSTM. LSTM will be unidirectional if omitted.'
    )
    parser.add_argument(
        '--dropout', type=float, default=0, help='Probability of applying dropout in final LSTM layer.'
    )
    parser.add_argument(
        '--batch_first', action='store_true', help='Assume batch size is first dimension of data.'
    )
    parser.add_argument(
        '--epochs', type=int, default=1, help='Number of passes through entire dataset during training.'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Adam optimizer learning rate.'
    )
    parser.add_argument(
        '--clip_grad', action='store_true',
        help = 'Apply gradient clipping. Set to True if included in command line arguments.'
    )
    parser.add_argument(
        '--max_norm', type=float, default=10.0
    )
    parser.add_argument(
        '-es', '--early_stopping_patience', type=int, default=3, help="Early stopping patience. Default: 3"
    )
    parser.add_argument(
        '--train_frac', type=float, default=0.75, help='Fraction of full dataset to separate for training.'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.15, help='Fraction of full dataset to separate for training.'
    )
    parser.add_argument(
        '--test_frac', type=float, default=0.10, help='Fraction of full dataset to separate for testing.'
    )
    parser.add_argument(
        '--subset_size', type=int, default=None,
        help='Number of datapoints to take as subset. If None, full dataset is taken.'
    )
    parser.add_argument(
        '--log_interval', type=int, default=5, help="Number of iterations between printing metrics."
    )
    parser.add_argument(
        '--no_tb', action='store_true', help='Turns off Tensorboard logging if included.'
    )
    parser.add_argument(
        '--w_loss', action='store_true', help='Applies frequency based weights to loss criterion '
                                              'if True (i.e., if included).'
    )
    parser.add_argument(
        '--w_sampling', action='store_true', help='Applies weighted random sampling '
                                                  'during training to deal with imbalanced data.'
    )
    # parser.add_argument('--padding_index', type=int, default=0,
    #                     help="Pos. int. value to use as padding when collating input batches.")

    # Parse command line arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Parse and print command line arguments for model configurations
    args = parse_arguments()

    print(f"Configuration: {args}")

    if args.mode == 'train':
        print("Starting training mode...")

        if not args.no_tb:
            # Create detailed experiment tag for tensorboard summary writer
            cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')

            log_dir = f'runs/{args.data}/lstm_emb_{args.embedding_dim}_hid_{args.hidden_dim}_l_{args.num_layers}_' \
                      f'bd_{args.bidirectional}_drop_{args.dropout}_bs_{args.batch_size}_epochs_{args.epochs}_' \
                      f'lr_{args.lr}_subset_{args.subset_size}_train_{args.train_frac}_val_{args.val_frac}_' \
                      f'test_{args.test_frac}_clip_{args.clip_grad}_maxnorm_{args.max_norm}_' \
                      f'es_{args.early_stopping_patience}_seed_{args.seed}_device_{args.device}_dt_{cur_datetime}'


            writer = SummaryWriter(log_dir)

            # Train model
            train(**vars(args), writer=writer)

            # close tensorboard summary writer
            writer.close()
        else:
            train(**vars(args))


    elif args.mode == 'val':
        print("Starting validation/development mode...")

        # hyper parameter search
        hp_search(**vars(args))

    elif args.mode == 'test':
        train(**vars(args))

    elif args.mode == 'tvt':
        train_test_val(**vars(args))
