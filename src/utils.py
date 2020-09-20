import os
import random
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


# -----------------------------------------------------------------------
# To check the data before collated into the batches:
def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# For metric calculation:
def get_eval_report(labels, preds):
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    
    return {
        'Accuracy': acc,
        'F1_Micro': f1_micro,
        'F1_Macro': f1_macro
    }
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# For metric calculation:
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# To save a checkpoint:
def save_checkpoint(checkpoints_path, model, mlp, tokenizer, optimizer, global_step, best_acc, epoch, best_checkpoint_path, name):

    # If the best checkpoint is being saved, the previous best checkpoint is deleted if it exists:
    if name == 'best':
        if best_checkpoint_path != None:
            shutil.rmtree(best_checkpoint_path)
        name = f'best-acc-{best_acc*100:.2f}-step-{global_step}'
        best_checkpoint_path = os.path.join(checkpoints_path, f'checkpoint-{name}')

    output_dir = os.path.join(checkpoints_path, f'checkpoint-{name}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(
        {'optimizer': optimizer.state_dict(),
        'mlp_state_dict': mlp.state_dict(),
        'global_step': global_step,
        'best_acc':  best_acc,
        'epoch':  epoch,
        'best_checkpoint_path': best_checkpoint_path,
        }, os.path.join(checkpoints_path, f'checkpoint-{name}', 'pytorch_aux.bin'))
    
    return best_checkpoint_path
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# To create the dataloaders:
def create_loaders(args, dataset, tokenizer):

    print('Creating the dataloaders...\n')
    # Loading the dataset:
    full_dataset = dataset(tokenizer, args)

    # Creating data indices for train and test splits:
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(args.test_split * dataset_size))
    train_indices = indices[split:]
    test_indices = indices[:split]

    # Creating the samplers and the loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(full_dataset, sampler=train_sampler, batch_size=args.batch_size_train, num_workers=args.num_workers, collate_fn=my_collate, worker_init_fn=lambda x: [np.random.seed((args.seed + x)), random.seed(args.seed + x), torch.manual_seed(args.seed + x)])
    test_dataloader = DataLoader(full_dataset, sampler=test_sampler, batch_size=args.batch_size_eval, num_workers=args.num_workers, collate_fn=my_collate, worker_init_fn=lambda x: [np.random.seed((args.seed + x)), random.seed(args.seed + x), torch.manual_seed(args.seed + x)])

    print(f'\nThere are {len(train_dataloader)} batches of {args.batch_size_train} sequences in the train loader.')
    print(f'There are {len(test_dataloader)} batches of {args.batch_size_eval} sequences in the test loader.\n')

    return train_dataloader, test_dataloader
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Create the directory structure:
def create_dir_structure(args):
    if not os.path.isdir('experiments'):
        os.mkdir('experiments')

    experiment_path = os.path.join('experiments', args.experiment_name)
    if os.path.isdir(experiment_path) and not args.resume:
        raise ValueError('Experiment directory already exists yet the training is not being resumed. Either use a different name for the experiment or delete the existing directory.')

    logs_path = os.path.join(experiment_path, 'logs')
    checkpoints_path = os.path.join(experiment_path, 'checkpoints')

    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)
        os.mkdir(logs_path)
        os.mkdir(checkpoints_path)

    return experiment_path, logs_path, checkpoints_path
# -----------------------------------------------------------------------
