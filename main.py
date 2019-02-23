import math
from tqdm import tqdm
import numpy as np
import torch
import time
import random
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import gc
import logging
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('rnn_attn.log')
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

# remove in kernel
from data_preprocess import load_and_prec, load_emb
from rnn_attn import rnn_attn

SEED = 1029
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(SEED)
## data preprocess and pre-trained embeddings

start_time = time.time()
tqdm.pandas()
train_X, train_y, test_X, word2idx = load_and_prec(maxlen=30, RAND_SEED=1029)
glove_file = '/home/panxie/Document/squad_qanet/data/original/Glove/glove.840B.300d.txt'
# para_file = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
embedding_matrix = load_emb(glove_file, word2idx)
# embedding_matrix_2 = load_emb(para_file, word2idx)
total_time = (time.time() - start_time) / 60
logger.info("Took {:.2f} minutes".format(total_time))
# embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)
# embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)

print(np.shape(embedding_matrix))
# del embedding_matrix_1, embedding_matrix_2
gc.collect()


# cross evalidation
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train_X, train_y))
# X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)
# search threshold
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold, pos_label=1)
        if score > best_score:
            best_threshold = threshold
            best_score = score
#     search_result = {'threshold': best_threshold, 'f1': best_score}
    return best_threshold, best_score

# config: parameters
class Config():
    batch_size = 512
    lr = 0.001
    lr_warm_up_num = 1000
    train_epochs = 5

args = Config()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train_preds = np.zeros((len(train_X)))
test_preds = np.zeros((len(test_X)))

seed_torch(SEED)

x_test_cuda = torch.LongTensor(test_X).cuda()
test = TensorDataset(x_test_cuda)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

for i, (train_idx, valid_idx) in enumerate(splits):
    x_train_fold = torch.LongTensor(train_X[train_idx]).cuda()
    y_train_fold = torch.FloatTensor(train_y[train_idx]).cuda()
    print("x_train_flod shape", x_train_fold.shape)
    print("y_train_fold shape", y_train_fold.shape)
    x_val_fold = torch.LongTensor(train_X[valid_idx]).cuda()
    y_val_fold = torch.FloatTensor(train_y[valid_idx]).cuda()
    print("x_val_flod shape", x_val_fold.shape)  # [261224, 76]
    print("y_val_fold shape", y_val_fold.shape)  # [261224,1]

    train = TensorDataset(x_train_fold, y_train_fold)
    valid = TensorDataset(x_val_fold, y_val_fold)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)

    model = rnn_attn(embedding_matrix, num_penultmate=16)
    logger.info(model)
    model.cuda()

    # set optimizer and scheduler
    param_group1 = []
    param_group2 = []
    for param in model.parameters():
        pass
    parameters_filter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        params=parameters_filter,
        lr=args.lr,
        betas=(0.8, 0.999),
        eps=1e-8,
        weight_decay=3e-7)
    cr = 1.0 / math.log(args.lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log(ee + 1)
        if ee < args.lr_warm_up_num else 1)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
    # optimizer = torch.optim.Adam(model.parameters())
    logger.info(f'Fold {i + 1}')

    valid_preds_fold = np.zeros((x_val_fold.size(0)))
    # threshold = 0.35
    for epoch in range(args.train_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()
        avg_loss /= len(train_loader)

        model.eval()
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item()
            valid_preds_fold[i * args.batch_size: (i + 1) * args.batch_size] = sigmoid(y_pred.cpu().numpy())
        threshold_fold, f1_fold = threshold_search(y_true=y_val_fold.cpu().numpy(), y_proba=valid_preds_fold)
        avg_val_loss /= len(valid_loader)
        elapsed_time = time.time() - start_time
        logger.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t threshold={:.4f} \t f1={:.4f} \t time={:.2f}s'.format(
            epoch + 1, args.train_epochs, avg_loss, avg_val_loss, threshold_fold, f1_fold, elapsed_time))


    test_preds_fold = np.zeros(len(test_X))
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()
        test_preds_fold[i * args.batch_size:(i + 1) * args.batch_size] = sigmoid(y_pred.cpu().numpy())

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold

test_preds /= len(splits)

res_threshold, res_score = threshold_search(train_y, train_preds)
logger.info("result threshold {} and result socre {}".format(res_threshold, res_score))