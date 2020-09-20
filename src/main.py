import logging
import os
import random
import sys
import time
from datetime import datetime
from dataset import Ranking_Dataset

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import (AlbertConfig, AlbertTokenizer, BertConfig,
                          BertTokenizer, GPT2Config, GPT2Tokenizer,
                          RobertaConfig, RobertaTokenizer)

from arguments import Arguments
from models.albert import AlbertForSequenceRanking
from models.bert import BertForSequenceRanking
from models.gpt2 import GPT2ForSequenceRanking
from models.mlp import Context_MLP
from models.roberta import RobertaForSequenceRanking
from utils import (compute_metrics, create_dir_structure, create_loaders,
                   save_checkpoint)

#-----------------------------------------
# Turning off logging from the external modules:
logging.getLogger("imported_module").setLevel(logging.ERROR)
#-----------------------------------------

#-----------------------------------------
# Parsing the arguments:
args = Arguments().parse()
#-----------------------------------------

#-----------------------------------------
# Setting device to CPU or GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------------------------------

#-----------------------------------------
# Setting the random seed:
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
#-----------------------------------------

#-----------------------------------------
# Creating the directory structure:
experiment_path, logs_path, checkpoints_path = create_dir_structure(args)
output_eval_file = os.path.join(experiment_path, f'results_{args.experiment_name}.txt')
#-----------------------------------------

#-----------------------------------------
# Preparing the config, tokeniser and the model classes:
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceRanking, BertTokenizer),
    'gpt2': (GPT2Config, GPT2ForSequenceRanking, GPT2Tokenizer),
    'albert': (AlbertConfig, AlbertForSequenceRanking, AlbertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceRanking, RobertaTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.model_name, num_labels=3, finetuning_task='classification')

if args.model_type in ['gpt2', 'albert']:
    config.summary_type = 'last'
    config.summary_use_proj=False
#-----------------------------------------

#-----------------------------------------
# Possible resuming from a checkpoint:
aux_checkpoint = None

if not args.resume:
    # Starting a nice fresh clean model:
    model = model_class.from_pretrained(args.model_name, config=config)
    tokenizer = tokenizer_class.from_pretrained(args.model_name,  do_lower_case=True)

else:
    # Resuming from a checkpoint:
    if args.resume_from == 'best':
        # Resuming from the best checkpoint:
        for dirpath, dirnames, filenames in os.walk(checkpoints_path):
            for name in dirnames:
                if name.startswith('checkpoint-best'):
                    resume_from = name.split('checkpoint-')[-1]
    else:
        # Resuming from a numbered or the last checkpoint:
        resume_from = args.resume_from

    checkpoints = f"{checkpoints_path}/checkpoint-{resume_from}"
    print(f'Loading from checkpoint {checkpoints}.\n')

    # Loading the model and the tokeniser:
    model = model_class.from_pretrained(checkpoints, config=config)
    tokenizer = tokenizer_class.from_pretrained(checkpoints)

    # Loading the auxiliary checkpoint, which keeps all non-HuggingFace components. The contents will be loaded later:
    aux_checkpoint = torch.load(os.path.join(checkpoints, 'pytorch_aux.bin'))
#-----------------------------------------

#-----------------------------------------
# Setting up the model:
model.to(device)

if args.model_type in ['bert', 'albert', 'roberta']:
    model_dim = config.hidden_size
elif args.model_type in ['gpt2']:
    model_dim = config.n_embd

mlp = Context_MLP(in_size=model_dim)
mlp = mlp.to(device)
#-----------------------------------------

#-----------------------------------------
# The loss function:
criterion = nn.MarginRankingLoss(margin=args.loss_margin, reduction='none')
#-----------------------------------------

#-----------------------------------------
# Creating the data loaders:
train_dataloader, test_dataloader = create_loaders(args, Ranking_Dataset, tokenizer)
#-----------------------------------------

#-----------------------------------------
# Tensorboard writer:
tb_writer = SummaryWriter(log_dir=f'{logs_path}/{datetime.now().strftime("%d%m%Y-%H_%M_%S")}/')
#-----------------------------------------

#-----------------------------------------
# Creating the optimiser:
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': mlp.parameters(), 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#-----------------------------------------

#-----------------------------------------
# Loading the contents of the auxiliary checkpoint and instantiating the contents if not resuming:
if aux_checkpoint:
    global_step = aux_checkpoint['global_step']
    epoch = aux_checkpoint['epoch']
    optimizer.load_state_dict(aux_checkpoint['optimizer'])
    best_acc = aux_checkpoint['best_acc']
    mlp.load_state_dict(aux_checkpoint['mlp_state_dict'])
    best_checkpoint_path = aux_checkpoint['best_checkpoint_path']
else:
    global_step = 0
    best_acc = 0.0
    epoch = 0
    best_checkpoint_path = None
#-----------------------------------------

#-----------------------------------------
# Enabling the use of dataparallel for multiple GPUs:
if args.dataparallel:
    model = nn.DataParallel(model)
#-----------------------------------------

#-----------------------------------------
# Preparing the calculation the pair label accuracy during training:
preds = None
out_label_ids = None
#-----------------------------------------

#-----------------------------------------
# Zero gradding the model:
model.zero_grad()
mlp.zero_grad()
#-----------------------------------------

#-----------------------------------------
# Measuring time:
start_time = time.time()
#-----------------------------------------
        
#-----------------------------------------
# The main training loop:
print('Beginning the training loop!!\n')
while epoch <= args.num_train_epochs:
    epoch += 1
    for _, batch in enumerate(train_dataloader):

        model.train()
        mlp.train()

        #---------------------------------
        # Getting the data out of the batch:
        left_tokens = batch['left_tokens'] 
        right_tokens = batch['right_tokens'] 
        label = batch['label']

        left_input_ids = left_tokens['input_ids'].to(device)
        left_attention_mask = left_tokens['attention_mask'].to(device)

        right_input_ids = right_tokens['input_ids'].to(device)
        right_attention_mask = right_tokens['attention_mask'].to(device)

        label = label.to(device)
        # Correcting the tensor dimensions for loss calculation:
        label = label.unsqueeze(dim=1)
        #---------------------------------

        #---------------------------------
        # Forward pass:
        left_rank_vec = model(left_input_ids, attention_mask=left_attention_mask)
        right_rank_vec = model(right_input_ids, attention_mask=right_attention_mask)
        left_rank_val, right_rank_val = mlp(left_rank_vec, right_rank_vec)
        #---------------------------------

        #---------------------------------
        # Calculating the loss:
        loss = criterion(left_rank_val, right_rank_val, label)
        loss = loss.mean()
        #---------------------------------

        #---------------------------------
        # Getting an accuracy score for pair labels during training:
        if preds is None:
            preds = left_rank_val > right_rank_val
            preds = preds.type(torch.int).detach().cpu().numpy()
            preds = [item for sublist in preds for item in sublist]
            preds = np.asarray(preds)
            out_label_ids = batch['label'].detach().cpu().numpy()
            out_label_ids[out_label_ids == -1] = 0
            out_label_ids = np.asarray(out_label_ids)
        else:
            temp_preds = left_rank_val > right_rank_val
            temp_preds = temp_preds.type(torch.int).detach().cpu().numpy()
            temp_preds = [item for sublist in temp_preds for item in sublist]
            temp_preds = np.asarray(temp_preds)
            preds = np.append(preds, temp_preds, axis=0)
            temp_out_label_ids = batch['label'].detach().cpu().numpy()
            temp_out_label_ids[temp_out_label_ids == -1] = 0
            temp_out_label_ids = np.asarray(temp_out_label_ids)
            out_label_ids = np.append(out_label_ids, temp_out_label_ids, axis=0)
        #---------------------------------

        #---------------------------------
        # Backward pass:
        if args.dataparallel:
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        #---------------------------------

        # #---------------------------------
        # Taking an optimisation step:
        optimizer.step()
        model.zero_grad()
        mlp.zero_grad()
        global_step += 1
        #---------------------------------

        #---------------------------------
        # Logging to tensorboard and standard output:
        if args.logging_freq > 0 and global_step % args.logging_freq == 0:

            # Stopping the clock to measure time:
            cur_time = time.time() - start_time
            # Calculating the training accuracy:
            training_result = compute_metrics(preds, out_label_ids)
            # Needed for the training accuracy:
            preds = None
            out_label_ids = None

            # Tensorboard logging:
            tb_writer.add_scalar('train/Accuracy', training_result['Accuracy'], global_step)
            tb_writer.add_scalar('train/F1_Micro', training_result['F1_Micro'], global_step)
            tb_writer.add_scalar('train/F1_Macro', training_result['F1_Macro'], global_step)
            tb_writer.add_scalar('train/Loss', loss.item(), global_step)
            acc = training_result['Accuracy']
            # Writing logs to standard output and tensorboard:
            print(f'Step: {global_step}\tLoss: {loss.item():0.3f}\tAccuracy: {acc:0.3f}\t\tTime/Batch : {cur_time/args.logging_freq:0.2f} seconds')

            if not (args.eval_freq > 0 and global_step % args.eval_freq == 0):
                start_time = time.time()
        #---------------------------------

        #---------------------------------
        # Running the eval loop:
        if args.eval_freq > 0 and global_step % args.eval_freq == 0: 
            print(f'\n------------Evaluating-------------')
            print(f'Step: {global_step}')

            #-----------------------------
            # Preparing for evaluation:
            eval_results = {}
            eval_loss = 0.0
            eval_preds = None
            eval_out_label_ids = None

            model.eval()
            mlp.eval()

            eval_start_time = time.time()
            #-----------------------------

            #-----------------------------
            # Beginning the evaluation loop:
            for eval_batch_num, eval_batch in enumerate(test_dataloader):
                with torch.no_grad():

                    #---------------------
                    # Loading eval data:

                    eval_left_tokens = eval_batch['left_tokens'] 
                    eval_right_tokens = eval_batch['right_tokens'] 
                    eval_label = eval_batch['label']

                    eval_left_input_ids = eval_left_tokens['input_ids'].to(device)
                    eval_left_attention_mask = eval_left_tokens['attention_mask'].to(device)

                    eval_right_input_ids = eval_right_tokens['input_ids'].to(device)
                    eval_right_attention_mask = eval_right_tokens['attention_mask'].to(device)

                    eval_label = eval_label.to(device)
                    # Correcting the tensor dimensions for loss calculation:
                    eval_label = eval_label.unsqueeze(dim=1)
                    #---------------------

                    #---------------------
                    # Forward pass for evaluation:
                    eval_left_rank_vec = model(eval_left_input_ids, attention_mask=eval_left_attention_mask)
                    eval_right_rank_vec = model(eval_right_input_ids, attention_mask=eval_right_attention_mask)
                    eval_left_rank_val, eval_right_rank_val = mlp(eval_left_rank_vec, eval_right_rank_vec)
                    #---------------------

                    #---------------------
                    # Calculating loss for evaluation step:
                    step_eval_loss = criterion(eval_left_rank_val, eval_right_rank_val, eval_label)
                    eval_loss += step_eval_loss.mean().item()
                    #---------------------

                    #---------------------
                    # Accuracy calculation during evaluation:
                    if eval_preds is None:   
                        eval_preds = eval_left_rank_val > eval_right_rank_val
                        eval_preds = eval_preds.type(torch.int).detach().cpu().numpy()
                        eval_preds = [item for sublist in eval_preds for item in sublist]
                        eval_preds = np.asarray(eval_preds)
                        eval_out_label_ids = eval_batch['label'].detach().cpu().numpy()
                        eval_out_label_ids[eval_out_label_ids == -1] = 0
                        eval_out_label_ids = np.asarray(eval_out_label_ids)
                    else:
                        temp_eval_preds = eval_left_rank_val > eval_right_rank_val
                        temp_eval_preds = temp_eval_preds.type(torch.int).detach().cpu().numpy()
                        temp_eval_preds = [item for sublist in temp_eval_preds for item in sublist]
                        temp_eval_preds = np.asarray(temp_eval_preds)
                        eval_preds = np.append(eval_preds, temp_eval_preds, axis=0)
                        temp_eval_out_label_ids = eval_batch['label'].detach().cpu().numpy()
                        temp_eval_out_label_ids[temp_eval_out_label_ids == -1] = 0
                        temp_eval_out_label_ids = np.asarray(temp_eval_out_label_ids)
                        eval_out_label_ids = np.append(eval_out_label_ids, temp_eval_out_label_ids, axis=0)
                    #---------------------
            # End of the evaluation loop
            #-----------------------------

            #-----------------------------
            # Calculating metrics:
            eval_loss = eval_loss / len(test_dataloader)
            eval_result = compute_metrics(eval_preds, eval_out_label_ids)
            eval_results.update(eval_result)
            eval_results['Loss'] = eval_loss
            #-----------------------------

            #-----------------------------
            # Writing evaluation results to the text file:
            with open(output_eval_file, 'a') as writer:
                writer.write(f'\n------------Evaluation Results -------------\n')
                writer.write(f'Step: {global_step}\n')
                for key in sorted(eval_results.keys()):
                    writer.write(f'{key} = {str(eval_results[key])}\n')
                writer.write(f'--------------------------------------------\n')
            #-----------------------------

            #-----------------------------
            # Logging evaluation results to standard output and tensorboard:
            txt = ''
            for key, value in eval_results.items():
                tb_writer.add_scalar('eval/{}'.format(key), value, global_step)
                txt += f'{key}: {value:0.3f}\t\t'
            print(txt)
            curr_time = (time.time() - eval_start_time)
            if curr_time < 60:
                unit = 'seconds'
            else:
                curr_time /= 60
                unit = 'minutes'
            print(f'Overall Evaluation Time: {curr_time:0.2f} {unit}')
            #-----------------------------

            #-----------------------------
            # Saving checkpoints if better evaluation accuracy is obtained:
            if eval_results['Accuracy'] > best_acc:
                best_acc = eval_results['Accuracy']
                best_checkpoint_path = save_checkpoint(checkpoints_path, model, mlp, tokenizer, optimizer, global_step, best_acc, epoch, best_checkpoint_path, 'best')
                best_checkpoint_path = save_checkpoint(checkpoints_path, model, mlp, tokenizer, optimizer, global_step, best_acc, epoch, best_checkpoint_path, 'last')
                print(f'Best accuracy checkpoint saved to best-acc-{best_acc*100:.2f}-step-{global_step}.')
            print(f'-----------------------------------\n')
            #-----------------------------

            #-----------------------------
            # Getting back to the training mode:
            model.train()
            mlp.train()
            start_time = time.time()
            #-----------------------------
        #---------------------------------

        #---------------------------------
        # Saving a checkpoint:
        if args.checkpointing_freq > 0 and global_step % args.checkpointing_freq == 0:
            best_checkpoint_path = save_checkpoint(checkpoints_path, model, mlp, tokenizer, optimizer, global_step, best_acc, epoch, best_checkpoint_path, global_step)
            best_checkpoint_path = save_checkpoint(checkpoints_path, model, mlp, tokenizer, optimizer, global_step, best_acc, epoch, best_checkpoint_path, 'last')
            print(f'\nCheckpoint saved at step {global_step}.\n')
        #---------------------------------
    #-------------------------------------

    #-------------------------------------
    print('\n***** End of an Epoch *****\n')
    #-------------------------------------
