import os
import torch
import torch.nn as nn
import random
from tqdm import tqdm, trange
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from data_preprocess import MrpcProcessor, convert_examples_to_features, logger, QuoraProcessor
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForMultipleChoice
from pytorch_pretrained_bert import BertForSequenceClassification, BertConfig, BertAdam
from sklearn.metrics import f1_score, precision_score, recall_score

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def compute_accuracy(output, labels):
    # outputs = np.argmax(out, axis=1)
    # print(outputs)
    # print(labels)
    return np.mean(output == labels)

def compute_f1_precision_recall(output, labels):
    # output = np.argmax(out, axis=1)
    # print(output[:20])
    # print(labels[:20])
    f1 = f1_score(y_true=labels, y_pred=output, pos_label=1)
    precision = precision_score(y_true=labels, y_pred=output, pos_label=1)
    recall = recall_score(y_true=labels, y_pred=output, pos_label=1)
    return f1, precision, recall

parser= argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
#parser.add_argument("--task_name",
#                    default=None,
#                    type=str,
#                    required=True,
#                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--max_seq_length",
                    default=30,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=False,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case",
                    default=False,
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--threshold',
                    type=float,
                    default=0.35,
                    help="higher than threshold is predicted to be positive example")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")

args = parser.parse_args()

class Bce_model(nn.Module):
    def __init__(self):
        super(Bce_model, self).__init__()
        self.bert = BertForMultipleChoice.from_pretrained("./pre_trained_models/bert-base-uncased.tar.gz", num_choices=1)
        self.loss = nn.BCELoss()

    def forward(self, input_ids, segment_ids, input_mask, label_ids):
        logits = self.bert(input_ids, segment_ids, input_mask)   # [batch, 1]
        out = torch.sigmoid(logits.squeeze(1))                                # [batch]
        loss = self.loss(out, label_ids.float())
        return out, loss          # [batch]

def main(debug=True):
    # device
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    # distribution training
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # gradient accumulation
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # use the same random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # output dir is need
    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    #os.makedirs(args.output_dir, exist_ok=True)

    # data processor
    processor  = QuoraProcessor()
    # processor = MrpcProcessor()
    num_labels = 2
    label_list = processor.get_labels()

    # BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained("./pre_trained_models/bert-base-uncased-vocab.txt")
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, debug=debug, debug_length=100)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    print(len(train_examples))

    model = Bce_model()

    if args.fp16:
        model.half()
    model.to(device)

    # distributed parallel
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # print(param_optimizer)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    def train(global_step, epoch):
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        total_loss, total_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # 加上 label_ids 输出结果是 loss
            # loss = model(input_ids, segment_ids, input_mask, label_ids)
            out, loss = model(input_ids, segment_ids, input_mask, label_ids)          # [batch, 1]
            logits = (out > args.threshold).type(torch.LongTensor)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_train_accuracy = compute_accuracy(logits, label_ids)
            # print(tmp_train_accuracy)

            total_loss += loss.mean().item()
            total_accuracy += tmp_train_accuracy

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # tr_loss += loss.item()
            # nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step/(t_total), args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            args.log_interval = 300
            if (step + 1) % args.log_interval == 0:
                # print(out)
                # print(pred)
                # print(label)
                logger.info("|----epoch {}, eclipse {}/{}, lr {:.4f},"
                            "loss {:.4f}, acc {:.4f}".format(
                    epoch, step + 1, len(train_dataloader), lr_this_step,
                    total_loss / args.log_interval, total_accuracy / args.log_interval))
                total_loss, total_accuracy = 0.0, 0.0


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # get dev examples anf dev features
        eval_examples = processor.get_dev_examples(args.data_dir, debug=debug, debug_length=16)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


    def eval():
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        out_all = []
        labels_all = []
        for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                out, tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            # print(out.shape)
            out_all.append(out)
            labels_all.append(label_ids)

        eval_loss = eval_loss / nb_eval_steps
        model.train()
        return eval_loss, out_all, labels_all

    if args.do_train and args.do_eval:
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            global_step = 0
            train(global_step, epoch)
            best_acc, best_f1 = 0.0, 0.0
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_loss, out_all, labels_all = eval()
                out_all = torch.cat(out_all, dim=0)
                labels_all = torch.cat(labels_all, dim=0)
                labels_eval = labels_all.cpu().numpy()
                for threshold in np.linspace(0.2, 0.6, 40):
                    logits_eval = (out_all > threshold).type(torch.LongTensor)
                    logits_eval = logits_eval.detach().cpu().numpy()
                    eval_accuracy = compute_accuracy(logits_eval, labels_eval)
                    eval_f1, eval_precision, eval_recall = compute_f1_precision_recall(logits_eval, labels_eval)
                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info('epcoh {:d}, threshold {:.4f}, accuracy {:.4f}, precision {:.4f}, recall {:.4f}, f1 {:.4f}, best_f1 {:.4f}'
                        .format(epoch, threshold, eval_accuracy, eval_precision, eval_recall, eval_f1, best_f1))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.do_train:
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
        model.to(device)
        eval_loss, out_all, labels_all = eval()
        # result = {'eval_loss': eval_loss,
        #           'eval_accuracy': eval_acc,
        #           "eval_f1": eval_f1}

        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results *****")
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main(debug=False)
