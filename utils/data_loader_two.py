import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import warnings
import torch
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
ItemValue = list()
# train_all = list()
train_user_set_p = defaultdict(list)
train_user_set_c = defaultdict(list)
train_user_set_v = defaultdict(list)
train_user_set_all = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        if 'None' in tmps:
            continue
        else:
            inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(train_data_p, train_data_c, train_data_v, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data_p[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data_p[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data_p:
        train_user_set_p[int(u_id)].append(int(i_id))
    for u_id, i_id in train_data_c:
        train_user_set_c[int(u_id)].append(int(i_id))
    for u_id, i_id in train_data_v:
        train_user_set_v[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))


def build_sparse_graph(data_cf, temp=0):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]
    if temp == 0:
        vals = [1.] * len(cf_)
    else:
        ItemValue.extend(ItemValue)
        vals = ItemValue
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items), dtype=float)
    return _bi_norm_lap(mat)


# 合并辅助信息矩阵， c，v
def build_sparse_graph_all(TrainPath, CartPath, PvPath, type_num, test_cf):
    with open(TrainPath, "r") as T:
        TrainDict = {}
        for line in T:
            DictVal = []
            ListVal = line.strip().split(' ')
            DictVal.append(ListVal[1:])
            DictVal.append([1.] * len(ListVal[1:]))
            TrainDict[ListVal[0]] = DictVal
    T.close()

    with open(CartPath, "r") as C:
        CartDict = {}
        for line in C:
            DictVal = []
            ListVal = line.strip().split(' ')
            DictVal.append(ListVal[1:])
            DictVal.append([1.] * len(ListVal[1:]))
            CartDict[ListVal[0]] = DictVal
    C.close()

    with open(PvPath, "r") as P:
        PvDict = {}
        for line in P:
            DictVal = []
            ListVal = line.strip().split(' ')
            DictVal.append(ListVal[1:])
            DictVal.append([1.] * len(ListVal[1:]))
            PvDict[ListVal[0]] = DictVal
    P.close()

    i = 0
    while i < n_users:
        TrainItem = TrainDict[str(i)]
        if str(i) in PvDict.keys():
           PvItem = PvDict[str(i)]
           for PItem in PvItem[0]:
                if PItem not in TrainItem[0]:
                   TrainItem[0].append(PItem)
                   TrainItem[1].append(type_num[i][1])
        TrainDict[str(i)] = TrainItem
        train_user_set_all[i] = TrainItem[0]
        for i_all in TrainItem[0]:
            if test_cf[i][1] == int(i_all):
                continue
        i = i + 1
    inter_mat = list()
    for Ukey in TrainDict.keys():
        IVal = TrainDict[Ukey]
        ItemValue.extend(IVal[1])
        for ite in IVal[0]:
            inter_mat.append([int(Ukey), int(ite)])
    return np.array(inter_mat)


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset

    print('reading train and test user-item set ...')
    directory = args.data_path + dataset + '/'
    trainPath = directory + 'train.txt'
    cartPath = directory + 'cart.txt'
    pvPath = directory + 'pv.txt'
    if args.dataset == "Yelp":
        train_p = read_cf(directory + 'trn_pos.txt')
        train_c = read_cf(directory + 'trn_neutral.txt')
        train_v = read_cf(directory + 'trn_tip.txt')
        # test_cf = (test_cf != None)
    else:
        train_p = read_cf(trainPath)
        train_c = read_cf(cartPath)
        train_v = read_cf(pvPath)
    test_cf = read_cf(directory + 'test.txt')
    valid_cf = test_cf
    pkfile = open(directory + "type_num_reset.txt", 'rb+')
    type_num_two = pickle.load(pkfile)
    pkfile = open(directory + "type_num.txt", 'rb+')
    type_num = pickle.load(pkfile)
    statistics(train_p, train_c, train_v, valid_cf, test_cf)
    """ Beibei -> train_p : [[0 4397]
                             [0 1814]
                             ...
                             [21715 62]]
    """
    trainAll = build_sparse_graph_all(trainPath, cartPath, pvPath, type_num_two, test_cf)
    norm_mat_p = build_sparse_graph(train_p)
    norm_mat_c = build_sparse_graph(train_c)
    norm_mat_v = build_sparse_graph(train_v)
    norm_mat_all = build_sparse_graph(trainAll, 1)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    print(f"n_users:{n_users}, n_items:{n_items}")

    user_dict = {
        'train_user_set_p': train_user_set_p,
        'train_user_set_c': train_user_set_c,
        'train_user_set_v': train_user_set_v,
        'train_user_set_all': train_user_set_all,
        'valid_user_set': None,
        'test_user_set': test_user_set,
    }
    print('loading over ...')
    return train_p, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v, norm_mat_all, torch.tensor(type_num)
