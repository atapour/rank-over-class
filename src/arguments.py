import argparse


# -------------------------------------------
# Arguments class:
class Arguments():

    # ---------------------------------------
    def __init__(self):
        self.initialized = False
    # ---------------------------------------

    # ---------------------------------------
    def initialize(self, parser):

        parser.add_argument('experiment_name', type=str, help='Name of the experiment. This determines directory structure and logging.')
        parser.add_argument('--data_dir', type=str, default='data', help='Path to the training data directory.')
        parser.add_argument('--data_file', type=str, default='data.csv', help='Name of the CSV file used for training.')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for the pytorch dataloader.')
        parser.add_argument('--loss_margin', type=float, default=2.0,  help='The value for the margin in the loss function.')
        parser.add_argument('--score_margin', type=float, default=0.02, help='The minimum distance between the ground truth ranking scores for training.')
        parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'gpt2', 'albert', 'roberta'], help='The type of model used for training. This code supports a limited number but more is available in the HuggingFace Transformers library.')
        parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased', 'gpt2', 'albert-base-v1', 'albert-base-v2', 'roberta-base'], help='The name of the model used for training. This must match the args.model_type. This code supports a limited number but more is available in the HuggingFace library.')
        parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum length of the text sequence. Longer sequences are truncated.')
        parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training.')
        parser.add_argument('--batch_size_eval', type=int, default=96, help='Batch size for evaluation.')
        parser.add_argument('--dataparallel', action='store_true', help='Enables the use of dataparallel for multiple GPUs.')
        parser.add_argument('--test_split', type=float, default=0.1, help='The ratio of data split for testing.')
        parser.add_argument('--seed', type=int, default=1, help='Random seed.')
        parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
        parser.add_argument('--learning_rate', type=float, default=4e-6, help='Learning rate.')
        parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon value for the Adam optimiser.')
        parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Value for gradient clipping.')
        parser.add_argument('--logging_freq', type=int, default=50, help='How many steps before periodic logging to the standard output and TensorBoard takes place.')
        parser.add_argument('--eval_freq', type=int, default=500, help='How many steps before periodic evaluation takes place.')
        parser.add_argument('--checkpointing_freq', type=int, default=2000, help='How many steps before checkpointing takes place.')
        parser.add_argument('--resume', action='store_true', help='Enables resuming training from a checkpoint.')
        parser.add_argument('--resume_from', type=str, default='last', help='Which checkpoint the training is resumed from. You can input the number of the global steps identifying the checkpoint, or use the words best or last as these checkpoints are saved separately. For instance, --resume_from=500 or --resume_from=best or --resume_from=last. Must be used in tandem with --resume.')
        parser.add_argument('--reprocess_input_data', action='store_true', help='Enable reprocessing the input data and ignores any cached files.')

        self.initialized = True

        return parser
    # ---------------------------------------

    # ---------------------------------------
    def get_args(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()
    # ---------------------------------------

    # ---------------------------------------
    def print_args(self, args):

        txt = '\n'
        txt += '-------------------- Arguments --------------------\n'
        for k, v in sorted(vars(args).items()):

            comment = ''
            default = self.parser.get_default(k)

            if v != default:
                comment = '\t[default: %s]' % str(default)

            txt += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)

        txt += '----------------------- End -----------------------'
        txt += '\n'

        print(txt)
    # ---------------------------------------

    # ---------------------------------------
    def parse(self):

        args = self.get_args()
        self.print_args(args)
        self.args = args

        return self.args
    # ---------------------------------------
