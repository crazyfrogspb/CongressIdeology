import os.path as osp

import mlflow
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from congressideology.config import config, reddit_config
from congressideology.models.reddit.classifier import initialize_model
from congressideology.models.reddit.load_data import download_model, load_data


def evaluate(model, data, data_loaders, criterion, dataset_type='val'):
    model.eval()
    epoch_loss = 0
    accuracy = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loaders[dataset_type]):
            logits, energy = model(batch)
            loss = criterion(logits, batch['target'])
            epoch_loss += loss.item()

            _, max_ind = torch.max(logits, 1)
            equal = torch.eq(max_ind, batch['target'])
            correct = int(torch.sum(equal))
            accuracy += correct

    model.train()

    return epoch_loss / (i + 1), accuracy / len(data[dataset_type])


def train_epoch(model, criterion, optimizer_ins, scheduler, clipping_value, data, data_loaders):
    # train model for one epoch
    epoch_loss = 0
    for i, batch in enumerate(data_loaders['train']):
        model.train()
        optimizer_ins.zero_grad()
        logits, energy = model(batch)
        loss = criterion(logits, batch['target'])
        loss.backward()
        clip_grad_norm_(filter(lambda p: p.requires_grad,
                               model.parameters()), clipping_value)
        optimizer_ins.step()
        epoch_loss += loss.item()

        if i % reddit_config.logging_freq == 0:
            val_loss, val_acc = evaluate(model, data, data_loaders, criterion)
            if scheduler is not None:
                scheduler.step(val_loss)
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_acc', val_acc)

    return epoch_loss / (i + 1)


def finalize_run(best_acc, best_loss):    # finalize run
    mlflow.log_metric('best_loss', best_loss)
    mlflow.log_metric('best_acc', best_acc)


def initialize_optimizer(optimizer, learning_rate, model):
    if optimizer == 'adam':
        optimizer_ins = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    elif optimizer == 'sgd':
        optimizer_ins = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Option {optimizer} is not supported for optimizer')

    return optimizer_ins


def train_model(directory, embedding_size, hidden_size, num_layers,
                dropout, bidirectional, min_count,
                batch_size, learning_rate, optimizer, clipping_value,
                n_epochs, early_stopping, subsample, run_id, model_type,
                random_state):
    # main function for training the model
    training_parameters = locals()
    training_parameters['model_type'] = 'encoder_classifier'

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    data, data_loaders, label_encoder = load_data(
        directory, min_count, subsample, batch_size)
    num_classes = len(label_encoder.classes_)

    model = initialize_model(data['train'].lang.n_words, embedding_size,
                             hidden_size, num_layers, dropout, bidirectional,
                             num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ins = initialize_optimizer(optimizer, learning_rate, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ins, 'min', patience=reddit_config.decay_patience,
        factor=reddit_config.decay_factor)
    last_epoch = -1
    best_acc = 0.0
    best_loss = np.inf
    early_counter = 0

    if run_id is not None:
        download_model(run_id)
        checkpoint = torch.load(
            osp.join(config.model_dir, 'reddit', f'checkpoint_{run_id}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        last_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        early_counter = checkpoint['early_counter']
        optimizer_ins.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        data['train'].lang = checkpoint['lang']

    with mlflow.start_run(run_uuid=run_id):
        for par_name, par_value in training_parameters.items():
            mlflow.log_param(par_name, par_value)

        for epoch in range(last_epoch + 1, n_epochs):
            print(f'Fitting epoch {epoch}')
            if early_counter >= early_stopping:
                finalize_run(best_acc, best_loss)
                return 'success'

            train_loss = train_epoch(
                model, criterion, optimizer_ins, scheduler, clipping_value, data, data_loaders)

            mlflow.log_metric('train_loss_epoch', train_loss)
            val_loss, val_acc = evaluate(model, data, data_loaders, criterion)
            mlflow.log_metric('val_loss_epoch', val_loss)
            mlflow.log_metric('val_acc_epoch', val_acc)

            if val_acc >= best_acc:
                early_counter = 0
                best_loss = val_loss
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_ins.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'lang': data['train'].lang,
                    'early_counter': early_counter,
                    'best_acc': best_acc,
                    'best_loss': best_loss
                }, osp.join(config.model_dir, 'reddit', f'checkpoint_{mlflow.active_run()._info.run_uuid}.pth'))
            else:
                early_counter += 1

        return 'success'
