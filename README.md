# _Rank over Class: The Untapped Potential of Ranking in Natural Language Processing_

Requires an NVIDIA GPU, Python 3, [CUDA CuDNN](https://developer.nvidia.com/cudnn), [PyTorch](http://pytorch.org), [Transformers](https://huggingface.co/transformers/), [pandas](https://pandas.pydata.org/), [shutil](https://pypi.org/project/pytest-shutil/) and [scikit-learn](https://scikit-learn.org/stable/). Note that the code has been tested with [Transformers 3.0.2](https://pypi.org/project/transformers/3.0.2/), [Transformers 2.9.1](https://pypi.org/project/transformers/2.9.1/) and [Transformers 2.9.0](https://pypi.org/project/transformers/2.9.0/). The code might not work with other versions of the [Transformers](https://huggingface.co/transformers/) package or models with different pre-trained weights from different versions of the package as this package is very frequently and substantially updated. Also note that running the code with different versions of the package will yield different results.

<br>

![General Pipeline](https://github.com/atapour/rank-over-class/blob/master/imgs/pipeline.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;General Pipeline of the Approach

## The Approach:

_"Text classification has long been a staple in natural language processing with applications spanning across sentiment analysis, online content tagging, recommender systems and spam detection. However, text classification, by nature, suffers from a variety of issues stemming from dataset imbalance, text ambiguity, subjectivity and the lack of linguistic context in the data. In this paper, we explore the use of text ranking, commonly used in information retrieval, to carry out challenging classification-based tasks. We propose a novel end-to-end ranking approach consisting of a Transformer network responsible for producing representations for a pair of text sequences, which are in turn passed into a context aggregating network outputting ranking scores used to determine an ordering to the sequences based on some notion of relevance. We perform numerous experiments on publicly-available datasets and investigate the possibility of applying our ranking approach to certain problems often addressed using classification. In an experiment on a heavily-skewed sentiment analysis dataset, converting ranking results to classification labels yields an approximately 22% improvement over state-of-the-art text classification, demonstrating the efficacy of text ranking over text classification in certain scenarios."_

[[Atapour-Abarghouei, Bonner and McGough, 2021](https://arxiv.org/abs/2009.05160)]

---
---

## Instructions:

* An example dataset is provided in `data/data.csv`. This dataset is execrated from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange). The dataset consists of 20,000 posts. All posts are randomly selected answers to questions from _Ask Ubuntu_, _Cryptography_, _Data Science_, _Network Engineering_, _Unix & Linux_ and _Webmasters_ communities. Each post contains a unique `user_id`, `norm_score` (number of votes the answer has received normalised by the number of votes the corresponding question has received), `body` (text of the answer with HTML Tags removed) and `community`.

* When the data is loaded once, tokenised input features are cached in the same directory as the input data. This avoids reprocessing the input in subsequent runs. To ensure the data is reprocessed again, the `--reprocess_input_data` argument can be used.

* Input arguments are as follows:
    * `experiment_name` (required positional argument): Name of the experiment. This determines directory structure and logging.
    * `--data_dir` (default=data): Path to the training data directory.
    * `--data_file` (default=data.csv): Name of the CSV file used for training.
    * `--num_workers` (default=1): Number of workers for the pytorch data loader.
    * `--loss_margin` (default=2.0): The value for the margin in the loss function.
    * `--score_margin` (default=0.02): The minimum distance between the ground truth ranking scores for training.
    * `--model_type` (default=bert, choices=[bert, gpt2, albert, roberta]): The type of model used for training. This code supports [BERT](https://huggingface.co/transformers/model_doc/bert.html), [OpenAI GPT2](https://huggingface.co/transformers/model_doc/gpt2.html), [ALBERT](https://huggingface.co/transformers/model_doc/albert.html) and [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) but more is available in the [HuggingFace Transformers](https://huggingface.co/transformers/) library.
    * `--model_name` (default=bert-base-uncased, choices=[bert-base-uncased, gpt2, albert-base-v1, albert-base-v2, roberta-base]): The name of the model used for training. This must match `args.model_type`. This code supports a limited number of the most basic models but more can be found in the [HuggingFace Transformers](https://huggingface.co/transformers/) library.
    * `--batch_size_train` (default=64): Batch size for training.
    * `--batch_size_eval` (default=96): Batch size for evaluation.
    * `--dataparallel` (boolean flag action): Enables the use of [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html) for multiple GPUs.
    * `--test_split` (default=0.1): The ratio of data split for testing.
    * `--seed` (default=1111): Random seed.
    * `--num_train_epochs` (default=100): Number of training epochs.
    * `--weight_decay` (default=0): Weight decay.
    * `--learning_rate` (default=4e-6): Learning rate.
    * `--adam_epsilon` (default=1e-8): Epsilon value for the Adam optimiser.
    * `--max_grad_norm` (default=1.0): Value for gradient clipping.
    * `--logging_freq` (default=50): How many steps before periodic logging to the standard output and TensorBoard takes place.
    * `--eval_freq` (default=500): How many steps before periodic evaluation takes place.
    * `-checkpointing_freq` (default=5000): How many steps before checkpointing takes place.
    * `-resume` (boolean flag action): Enables resuming training from a checkpoint.
    * `--resume_from` (default=last): Which checkpoint the training is resumed from. You can input the number of the global steps identifying the checkpoint, or use the words 'best' or 'last' as these checkpoints are saved separately. For instance `--resume_from=500` or `--resume_from=best` or `--resume_from=last`. Must be used in tandem with `--resume`.
    * `--reprocess_input_data` (boolean flag action): Enable reprocessing the input data and ignores any cached files.

* The code utilizes [TensorBoard](https://www.tensorflow.org/tensorboard/) from [torch.utils](https://pytorch.org/docs/stable/tensorboard.html) to display plots for better analysis. The TensorBoard logs are saved at `experiments/<experiment_name>/logs`.

* A directory called `experiments` is created to hold information for all experiments. A directory with a name determined via the `experiment_name` argument is created inside the `experiments` directory. Checkpoints are saved in the `checkpoints` directory and the TensorBoard logs are saved in the `logs` directory. Additionally, he output of every evaluation loop is saved in the text file `experiments/<experiment_name>/results_<experiment_name>.txt`.

* To clone this repository, run the following command:

```
$ git clone https://github.com/atapour/rank-over-class.git
$ cd rank-over-class
```

* To train the model, run the following command:

```
$ python src/main.py <experiment_name> --data_dir=data --data_file=data.csv --loss_margin=2.0 --score_margin=0.02 --model_type=bert --model_name=bert-base-uncased --batch_size_train=64 --batch_size_eval=96 --test_split=0.1 --logging_freq=50 --eval_freq=500 --checkpointing_freq=5000
```

---

![Code Run Example](https://github.com/atapour/rank-over-class/blob/master/imgs/terminal.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;Example of How the Code Is Run

---

## Video Detailing the Approach:
[![Video Example](https://github.com/atapour/rank-over-class/blob/master/imgs/thumbnail.png)](https://youtu.be/5GLZ9zH_hao "Video Detailing the Approach - Click to Play")

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Video Detailing the Approach - click image above to play.

---

## Reference:

This work is created as part of the project published in the following:

[Rank over Class: Applications of Ranking in Natural Language Processing](https://arxiv.org/pdf/2009.05160.pdf)
(A. Atapour-Abarghouei, S. Bonner and A.S. McGough), 2021. [[pdf](http://www.atapour.co.uk/papers/RankOverClass.pdf)]

```

@article{atapour2021rankoverclass,
  title={Rank over Class: The Untapped Potential of Ranking in Natural Language Processing},
  author={Atapour-Abarghouei, Amir and Bonner, Stephen and McGough, Andrew Stephen},
  journal={arXiv preprint arXiv:2009.05160},
  year={2021}
}


```
---
