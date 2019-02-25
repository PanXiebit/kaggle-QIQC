# -*- encoding = utf-8 -*-

import csv
import pandas as pd
import os
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("quora_bert.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]   >> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger = get_logger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, debug=False, debug_length=3):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, debug=False, debug_length=3):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, debug=False, debug_length=3):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for i, line in enumerate(reader):
                if debug and i > debug_length:
                    break
                lines.append(line)
            return lines

    @classmethod
    # def _read_csv(cls, input_file, quotechar=None, debug=False, debug_length=3):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding='utf-8') as f:
    #         # reader = csv.reader(f, delimiter=",", quotechar=quotechar)
    #         reader = csv.reader(f, delimiter='|', quotechar=quotechar)
    #         lines = []
    #         for i, line in enumerate(reader):
    #             if debug and i > debug_length:
    #                 break
    #             lines.append(line)
    #         return lines
    def _read_csv(cls, input_file, debug=False, debug_length=3):
        df = pd.read_csv(input_file)
        lines = []
        for i in range(len(df)):
            line = []
            line.append(df["qid"][i])
            line.append(df["question_text"][i])
            line.append(df["target"][i])
            if len(line) != 3:
                print(line)
            lines.append(line)
        return lines

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, debug=False, debug_length=3):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), None, debug, debug_length), "train")

    def get_dev_examples(self, data_dir, debug=False, debug_length=3):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv"), None, debug, debug_length), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QuoraProcessor(DataProcessor):
    def get_train_examples(self, data_dir, debug=False, debug_length=3):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(os.path.join(data_dir, "train.csv"),"train", debug, debug_length)

    def get_dev_examples(self, data_dir, debug=False, debug_length=3):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "dev.csv"), "dev", debug, debug_length)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, filename, set_type, debug, debug_lenght):
        """Creates examples for the training and dev sets."""
        examples = []
        df_data = pd.read_csv(filename)
        for i in range(len(df_data)):
            if i > debug_lenght and debug:
                break
            guid = df_data["qid"][i]
            text_a = df_data["question_text"][i]
            text_b = None
            label = df_data["target"][i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    # print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # print(tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # print(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # print(tokens)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # print(type(example.label))   # int
        label_id = label_map[example.label]
        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

if __name__ == "__main__":
    # test processor
    # data_dir = "glue_data/MRPC"
    # processor = MrpcProcessor()
    # dev_examples = processor.get_dev_examples(data_dir, debug=True)
    # print(dev_examples[0].text_a)
    # print(dev_examples[0].text_b)
    # print(dev_examples[0].label)
    # print(processor.get_labels())
    # print(len(dev_examples))
    #
    # # test examples to feature
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # label_list = processor.get_labels()
    # dev_features = convert_examples_to_features(dev_examples, label_list, 100, tokenizer)

    # test quora classification
    data_dir = "/home/panxie/Document/kaggle/quora/data/splited_data/"
    processor = QuoraProcessor()
    dev_examples = processor.get_dev_examples(data_dir, debug=True)
    print(len(dev_examples))
    for i in range(len(dev_examples)):
        if dev_examples[i].label == 0:
            print(dev_examples[i].guid, dev_examples[i].text_a, dev_examples[i].label)


    # tokenizer = BertTokenizer.from_pretrained("/home/panxie/Document/GLUE/pre_trained_models/bert-base-uncased-vocab.txt")
    # label_list = processor.get_labels()
    # dev_features = convert_examples_to_features(dev_examples, label_list, 30, tokenizer)
    # print(len(dev_features))
    # print(dev_features[0])
