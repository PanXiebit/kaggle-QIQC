import re
import spacy
import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm


### data preprocess
NLP = spacy.blank("en")
def word_tokenize(sent):
    sent = sent.lower()
    doc = NLP(sent)
    return [token.text for token in doc]

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def get_wordCounter_and_word2idx(texts, limit=5):
    wordCounter = Counter()
    for text in texts:
        for token in text:
            wordCounter[token] += 1
    word2idx = {}
    special_tokens= {"<PAD>":0, "<OOV>":1, "<EOS>":2, "<SOS>":3}
    index = 4
    for token, count in wordCounter.items():
        if count <= limit:
            continue
        word2idx[token] = index
        index += 1
    word2idx.update(special_tokens)
    return wordCounter, word2idx

def get_index(word, word2idx):
    if word in word2idx:
        return word2idx.get(word)
    else:
        return word2idx.get("<OOV>")

def pad_sequence(texts, word2idx, maxlen=30):
    """
    texts: numpy.array (text_size,)
    """
    texts_feature = np.zeros((len(texts), maxlen))
    for i, text in enumerate(texts):
        if len(text) > maxlen:
            text = text[:maxlen]
        text_feature = [get_index(token, word2idx) for token in text]
        text_feature += [0] * (maxlen - len(text_feature))
        texts_feature[i] = np.array(text_feature)
    return texts_feature


def load_and_prec(maxlen=30, RAND_SEED=1029):
    train_df = pd.read_csv("/home/panxie/Document/kaggle/quora/data/train.csv")
    test_df = pd.read_csv("/home/panxie/Document/kaggle/quora/data/test.csv")
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # lower
    tqdm.pandas()
    # lower
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())

    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))

    # Clean speelings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    tqdm.pandas()
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: word_tokenize(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: word_tokenize(x))

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    wordCounter, word2idx = get_wordCounter_and_word2idx(train_X, limit=5)

    train_X = pad_sequence(train_X, word2idx, maxlen=maxlen)
    test_X = pad_sequence(test_X, word2idx, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values

    # shuffling the training data
    np.random.seed(RAND_SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]

    return train_X, train_y, test_X, word2idx

def load_emb(emb_file, word2idx, vec_size=300, debug=False):
    if emb_file is not None:
        print("load embedding from {}".format(emb_file))
        embedding = np.zeros((len(word2idx), vec_size), dtype=np.float32)
        count = 0
        with open(emb_file, encoding="utf8", errors='ignore') as f:
            for i, line in tqdm(enumerate(f)):
                if debug and i > 3:
                    break
                if len(line.split()) < 10:
                    print(line)
                    continue
                line = line.strip().split()
                word = "".join(line[:-vec_size])
                vec = np.array(list(float(x) for x in line[-vec_size:]))
                if word in word2idx:
                    count += 1
                    embedding[word2idx[word]] = vec
                else:
                    continue
            print("{}/{} have embedding from".format(count, len(word2idx)))
    else:
        print("Generate random embedding")
        embedding = np.random.uniform(-0.8, 0.8, (len(word2idx), vec_size))
    return embedding

if __name__ == "__main__":
    train_X, train_y, test_X, word2idx = load_and_prec(maxlen=30, RAND_SEED=1029)
    print(train_X.shape)