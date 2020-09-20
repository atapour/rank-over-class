import multiprocessing
import os
import random
import time
from itertools import combinations
from multiprocessing import Pool, cpu_count

import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------------------------------------
# The primary dataset class:
class Ranking_Dataset(Dataset):

    def __init__(self, tokenizer, args):

        # -----------------------------------------------
        # Random seed:
        random.seed(args.seed)
        # The score margin determines the distance between the score of the pair to be ranked during training:
        self.score_margin = args.score_margin
        # The tokeniser:
        self.tokenizer = tokenizer.from_pretrained(args.model_name)
        self.model_type = args.model_type
        if args.model_type == 'gpt2':
            self.tokenizer.pad_token = "[PAD]"
            self.add_tokens = True
        else:
            self.add_tokens = False

        self.tokenizer.padding_side = 'right'
        # Maximum sequence length:
        self.max_seq_length = args.max_seq_length
        # The processed data is saved and loaded if available to avoid re-processing everytime the code is run:
        cached_input_data = os.path.join(args.data_dir, f'cached_input_{os.path.splitext(args.data_file)[0]}')
        # -----------------------------------------------

        # -----------------------------------------------
        # Checking to see if the input data has already been processed and cached:
        if os.path.exists(cached_input_data) and not args.reprocess_input_data:

            print(f'Input data already processed. Reading from {cached_input_data}')

            input_dict = torch.load(cached_input_data)
            self.user_answer_pairs_row_ids = input_dict['user_answer_pairs_row_ids']
            self.big_data = input_dict['big_data']
        # -----------------------------------------------

        # -----------------------------------------------
        # The input data needs to be processed and then saved to disk when ready:
        else:
            csv_file_name = os.path.join(args.data_dir, args.data_file)

            print(f'Reading the csv file from {csv_file_name}...')
            self.big_data = pd.read_csv(csv_file_name, lineterminator='\n')
            self.big_data.dropna(inplace=True)
            self.big_data = self.big_data[self.big_data['body'].notna()]
            # -------------------------------------------

            # -------------------------------------------
            print('Cleaning the dataframe...', end = '')
            
            start = time.time()

            if 'community' in self.big_data:
                self.big_data = self.big_data.drop('community', axis=1)    
                
            self.big_data.reset_index(drop=True, inplace=True)

            print(f' {time.time() - start:.2f} seconds.')
            # -------------------------------------------

            # -------------------------------------------
            # Empty lists to store our data:
            self.user_answer_pairs_row_ids = []
            self.user_answer_pairs_row_ids_post_removal = []
            # -------------------------------------------

            # -------------------------------------------
            print('Getting all possible pair combinations...', end = '')

            start = time.time()
            combination_series = self.big_data.groupby('user_id').apply(lambda x: list(combinations(x.index, 2)))

            for id_pair_lists in combination_series:
                self.user_answer_pairs_row_ids.extend(id_pair_lists)

            print(f' {time.time() - start:.2f} seconds.')
            # -------------------------------------------

            # -------------------------------------------
            print('Shuffling the pairs...', end = '')

            start = time.time()
            random.shuffle(self.user_answer_pairs_row_ids)

            print(f' {time.time() - start:.2f} seconds.')
            # -------------------------------------------

            # -------------------------------------------
            print(f'Removing pairs with a score difference of less than {self.score_margin}...', end = '')

            start = time.time()
            # Available cores on the machine for parallel processing
            num_processes = multiprocessing.cpu_count()
            # Determining the number of chunks
            chunk_size = max([int(len(self.user_answer_pairs_row_ids)/num_processes), 1])
            # Creating the pool for multi processing
            pool = multiprocessing.Pool(processes=num_processes)
            # Dividing the list to chunks
            chunks = [self.user_answer_pairs_row_ids[i:i + chunk_size] for i in range(0, len(self.user_answer_pairs_row_ids), chunk_size)]
            # Applying the remover helper function to the chucks
            results = pool.map(self.bad_score_pair_remover_for_loop, chunks)
            # Getting the results in a new list
            for result in results:
                self.user_answer_pairs_row_ids_post_removal.extend(result)
            pool.close()
            print(f' {(time.time() - start):.3f} seconds.')
            # -------------------------------------------

            # -------------------------------------------
            # Removing any possible duplicates (just a precaution):
            self.user_answer_pairs_row_ids = list(set(self.user_answer_pairs_row_ids_post_removal))
            # -------------------------------------------

            # -------------------------------------------
            # Saving the processed input to disk as dictionary:
            output_dict = {'user_answer_pairs_row_ids': self.user_answer_pairs_row_ids, 'big_data': self.big_data}
            torch.save(output_dict, cached_input_data)
            print(f'Input data saved into cached file {cached_input_data}.')
            # -------------------------------------------
        # -----------------------------------------------

    # ---------------------------------------------------
    def __len__(self):
        return len(self.user_answer_pairs_row_ids)
    # ---------------------------------------------------

    # ---------------------------------------------------
    def __getitem__(self, idx):

        # -----------------------------------------------
        # Extracting a pair from the big_data based on the idx:
        left_id, right_id = self.user_answer_pairs_row_ids[idx]
        left_row = self.big_data.iloc[left_id] 
        left_score = left_row.norm_score
        right_row = self.big_data.iloc[right_id] 
        right_score = right_row.norm_score
        left_body = left_row['body']
        right_body = right_row['body']

        # Getting the tokens for the pair:
        left_tokens = self.tokenizer.encode_plus(left_body, add_special_tokens=self.add_tokens, truncation=True, max_length=self.max_seq_length, pad_to_max_length=True, return_tensors='pt')
        right_tokens = self.tokenizer.encode_plus(right_body, add_special_tokens=self.add_tokens, truncation=True, max_length=self.max_seq_length, pad_to_max_length=True, return_tensors='pt')

        # Reshaping the tokens as expected by models:
        left_tokens['input_ids'] = left_tokens['input_ids'].squeeze()
        right_tokens['input_ids'] = right_tokens['input_ids'].squeeze()
        if self.model_type == 'albert':
            left_tokens['attention_mask'] = left_tokens['attention_mask'].squeeze()
            right_tokens['attention_mask'] = right_tokens['attention_mask'].squeeze()

        # Checking to make sure the training pair do not have the same ranking score:
        if left_score == right_score:
            raise ValueError('A pair with equal ranking scores was encountered.')

        # Creating the ranking lable:
        label = -1 if left_score < right_score else 1
        label = torch.tensor(label, dtype=torch.long)
        
        return {'left_tokens': left_tokens, 'right_tokens': right_tokens, 'label': label}
    # ---------------------------------------------------

    # ---------------------------------------------------
    # Helper function to remove bad pairs:
    def bad_score_pair_remover_for_loop(self, pair_list):

        pair_list_without_close_scores = []
        for pair in pair_list:
            left_id, right_id = pair
            left_row = self.big_data.iloc[left_id]
            right_row = self.big_data.iloc[right_id]
            left_score = left_row.norm_score
            right_score = right_row.norm_score

            if abs(left_score - right_score) > self.score_margin:
                pair_list_without_close_scores.append(pair)

        return pair_list_without_close_scores
    # ---------------------------------------------------
# -------------------------------------------------------
