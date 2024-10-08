from glob import glob
from functools import reduce
from itertools import *
from more_itertools import *
from sklearn.metrics import *

import os, sys, random
import json, pickle
import torch
import numpy as np
import pandas as pd

# seed for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# set gpu
def set_gpu(gpu=0):
    torch.set_default_device('mps')
    #torch.cuda.set_device(torch.device('mps'))

# system logging
def eprint(s):  sys.stderr.write(s)

# put labels based on threshold: baseline
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

# best possible accuracy
def best_acc(score, labels):
    A = list(zip(labels, score))
    A = sorted(A, key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0]==1])
    tn = 0
    th_acc = []
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        th_acc.append((th, acc))
    return max(th_acc, key=lambda x: x[1])[1]

# best possible f1 score
def best_f1(score, labels):
    prec, rec, _ = precision_recall_curve(labels, score)
    f1 = 2*rec*prec/(rec+prec)
    return np.nanmax(f1)

# determine threshold maximizing each measure
def get_thresh(score, labels, maximize='auc'):
    if maximize == 'auc':
        q = 1 - sum(labels)/len(labels)
        return np.quantile(score, q)
    elif maximize == 'prec':
        s_zero = [s for s,l in zip(score,labels) if l==0]
        return max(s_zero) + 1e-10
    elif maximize == 'rec':
        s_one = [s for s,l in zip(score,labels) if l==1]
        return min(s_one) - 1e-10
    else:
        return eval(maximize+'_thresh')(score, labels)

# best possible accuracy
def acc_thresh(score, labels):
    A = list(zip(labels, score))
    A = sorted(A, key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0]==1])
    tn = 0
    th_acc = []
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        th_acc.append((th, acc))
    return max(th_acc, key=lambda x: x[1])[0]

# get auc from scores
def get_score(score, labels):
    return roc_auc_score(labels, score)

# get stats from preds and labels
def get_stats(preds, labels):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return (acc, prec, rec, f1)

# confusion matrix
def conf_matrix(preds, labels):
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(len(preds)):
        if labels[i]==preds[i]==1:
           TP += 1
        if preds[i]==1 and labels[i]!=preds[i]:
           FP += 1
        if labels[i]==preds[i]==0:
           TN += 1
        if preds[i]==0 and labels[i]!=preds[i]:
           FN += 1

    return(TP, TN, FP, FN)

# stats
def nanstats(scores):
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    else:
        return np.nanmean(scores), np.nanmax(scores), np.nanmin(scores)

# get voting results from player predictions
def get_votes(preds):
    l = len(preds)
    assert len(preds[0]) == l

    tot_votes = np.sum(preds, axis=0)               # total number of votes
    self_cast = preds[np.eye(l, dtype=bool)]        # self-casting vote
    res = (tot_votes - self_cast) / l >= 0.5        # voting result

    agr = np.maximum(tot_votes, l-tot_votes) / l    # agreements

    return res, agr

# get voting results under malicious attacks
def get_votes_attack(preds, teams, n_atk=1, attacker=None, seed=0):
    l = len(preds)
    assert len(preds[0]) == l
    assert n_atk >= 0
    assert attacker is not None

    if n_atk > l:
        n_atk = l

    # set seed
    np.random.seed(seed)

    # divide teams
    t_a = np.where(teams[0,:] == 0)[0]
    t_b = np.where(teams[0,:] == 1)[0]
    n_1 = int(n_atk/2)
    n_2 = n_atk - n_1

    # randomly select attackers
    if n_1 == n_2:
        atk_a = np.random.choice(t_a, size=n_1, replace=False)
        atk_b = np.random.choice(t_b, size=n_2, replace=False)
    elif np.random.uniform() >= 0.5:
        atk_a = np.random.choice(t_a, size=n_1, replace=False)
        atk_b = np.random.choice(t_b, size=n_2, replace=False)
    else:
        atk_a = np.random.choice(t_a, size=n_2, replace=False)
        atk_b = np.random.choice(t_b, size=n_1, replace=False)
    attackers = np.concatenate([atk_a, atk_b])

    # modify vote
    if attacker == 'dishonest-but-rational (all)':
        for a in attackers:
            preds[a,:] = abs(teams[a,:] - 1)
    elif attacker == 'dishonest-but-rational (allies)':
        for a in attackers:
            preds[a,:] = np.where(teams[a,:],np.zeros_like(teams[a,:]),preds[a,:])
    elif attacker == 'dishonest (flip)':
        for a in attackers:
            preds[a,:] = abs(preds[a,:] - 1)
    elif attacker == 'dishonest (random)':
        for a in attackers:
            preds[a,:] = np.random.randint(2, size=l)
    else:
        print('error!')
        exit()

    tot_votes = np.sum(preds, axis=0)               # total number of votes
    self_cast = preds[np.eye(l, dtype=bool)]        # self-casting vote
    res = (tot_votes - self_cast) / l >= 0.5        # voting result

    return res

# get median value for voting threshold
def get_median(scores):
    l = len(scores)
    assert len(scores[0]) == l

    median = []
    for i,c in enumerate(scores.T):
        c_except = c[np.arange(l)!=i]               # remove self-score
        med = np.median(c_except)
        median.append(med)

    return median

# benchmark all
def bench_all(args, stats):
    bench_file = 'bench_all.tsv'

    bench_all = '\t'.join(map(str, args)) + '\t'
    bench_all += '\t'.join(map(lambda i: f'{i:.4f}', stats))
    bench_all += '\n'

    with open(bench_file, 'a') as f:
        f.write(bench_all)


# get id from filename
def get_id(filename, ext=''):
    return filename.split('/')[-1].split('_')[-1].strip(f'.{ext}')


# get cheater list
def get_cheater(e, exp_prefix, data_dir='data_processed'):
    cheater = {}
    # retrieve cheater list from an external file
    with open(f'{data_dir}/{exp_prefix}/game_{game}/cheater', 'r') as f:
        cheater[game] = sorted([r.split(': ')[0] for r in f], key=int)


# process and load dataframes
def get_frames(exp_prefix, all_cols, data_cols, win_size, duration, rel='fire'):
    frames, cheater, times = {}, {}, {}
    cols = ['timestamp'] + all_cols + ['aimhack']
    col_id = 'src' if rel == 'fire' else 'dst'

    col_loc = ['loc_x', 'loc_y', 'loc_z']

    # for each game
    for dir_g in sorted(glob(f'{exp_prefix}/game_*')):
        g = get_id(dir_g)

        # cheater list
        with open(f'{dir_g}/cheater', 'r') as f:
            # modified key identifying expname
            g_ = '_'.join([get_id(exp_prefix), g])
            cheater[g_] = sorted([r.split(': ')[0] for r in f], key=int)

        # for each observer
        for dir_o in sorted(glob(f'{dir_g}/obs_*')):
            o = get_id(dir_o)

            # read relevant log files
            df_event = pd.read_csv(f'{dir_o}/log_event_processed.csv')
            df_rel = df_event[df_event['event'] == rel]

            times[g_] = (min(df_event['timestamp']), max(df_event['timestamp']))

            # for each player
            for log_p in sorted(glob(f'{dir_o}/log_player_*.csv')):
                p = get_id(log_p, ext='csv')     # player id

                if p.isdigit() and int(p) in set(df_event[col_id]):
                    # timestamps where the relevant event occured
                    ts_rel = df_rel[df_rel[col_id] == int(p)]['timestamp']

                    ts = map(lambda t: range(t-win_size-duration,t+duration), ts_rel)
                    ts_sorted = sorted(reduce(lambda x,y: set(x).union(y), ts))

                    df = pd.read_csv(log_p)
                    df['aimhack'] = [p in cheater[g_]]*len(df)

                    # filter dataframes that diff() exceeds some threshold
                    diff_df = df.diff()
                    dist_df = diff_df[col_loc].apply(np.linalg.norm, axis=1)
                    df = df[dist_df < 50]

                    # drop nan
                    dff = df[df['timestamp'].isin(ts_sorted)][cols].dropna(subset=data_cols)

                    key = (g_, o, p)
                    frames[key] = dff

    return frames, cheater, times


# smooth dataframes
def smooth_df(df, drop=['timestamp','aimhack']):
    cols = df.columns.drop(drop)
    df[cols] = df[cols].ewm(alpha=0.9).mean()
    return df

# normalize dataframe
def normalize(df, norm_args):
    ndf = df.copy()

    mi, mx = norm_args
    for c in df.columns:
        if mi[c] == mx[c]:
            ndf[c] = df[c] - mi[c]
        else:
            ndf[c] = (df[c] - mi[c]) / (mx[c] - mi[c])

    return ndf

# decide labels
def decide_label(ts, labels, d_th):
    return np.add.accumulate(labels) > d_th

# return closest element in array
def closest(v, l):
    l = np.asarray(l)
    return (np.abs(l-v)).argmin(), (np.abs(l-v)).min()

# helper function for identifying same team
def get_teams(exp_name, g):
    team_dir = f'data_processed/{exp_name}/game_{g}/teams'
    teams = []
    with open(team_dir, 'r') as f:
        for r in f.readlines():
            teams.append([a[-1] for a in r.replace('\n','').split(': ')][:-1])
    teams = [set(map(int,a)) for a in teams]
    return teams

# indicator for the same team
def team_indicator(teams):
    assert len(teams) == 2 and len(teams[0]) == len(teams[1])

    l = len(teams[0])
    ind = np.zeros((2*l,2*l))
    for i,j in product(teams[0], repeat=2):
        ind[i,j] = True
    for i,j in product(teams[1], repeat=2):
        ind[i,j] = True

    return ind

# evaluate prediction accuracies based on true labels and scores
def eval_preds(eval_trues, eval_scores, incl_cnt=False):
    """ Evaluate prediction accuracies for given true labels and scores

    Parameters
    ----------
    eval_trues: dictionary of true labels for each game
    eval_scores: dictionary of scores for each game
    incl_cnt: include absolute counts

    Returns
    ----------
    b_acc: accuracy at best accuracy
    b_prec: accuracy at best precision
    auc: area under the roc curve score
    """

    # placeholder for aggregated predictions
    p_agg_prec, p_agg_acc = [], []

    # aggregated true labels and boundary error scores
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([get_median(s) for s in eval_scores.values()], [])

    # determine threshold based on scores and labels
    thresh_prec = get_thresh(s_agg, t_agg, maximize='prec')
    thresh_acc = get_thresh(s_agg, t_agg, maximize='acc')

    # predictions based on scores and threshold
    for scores in eval_scores.values():
        preds_prec = scores > thresh_prec
        preds_acc = scores > thresh_acc
        p_prec,_ = get_votes(preds_prec)
        p_acc,_ = get_votes(preds_acc)

        p_agg_prec.extend(p_prec)
        p_agg_acc.extend(p_acc)

    # evaluation metrics: best_acc, best_prec, auc_roc
    b_acc = best_acc(s_agg, t_agg)
    b_prec = get_stats(p_agg_prec, t_agg)[0]
    auc = roc_auc_score(t_agg, s_agg)

    # confusion matrix under best accuracy threshold
    conf_mat = conf_matrix(p_agg_acc, t_agg)

    if incl_cnt:
        return b_acc, b_prec, auc, len(p_agg_prec), *conf_mat
    else:
        return b_acc, b_prec, auc, len(p_agg_prec)


# evaluate prediction accuracies based on true labels and vote scores
def eval_vote_preds_normal(eval_trues, vote_scores, incl_cnt=False):
    """ Evaluate prediction accuracies for given true labels and scores

    Parameters
    ----------
    eval_trues: dictionary of true labels for each game
    vote_scores: dictionary of final scores for voting
    incl_cnt: include absolute counts

    Returns
    ----------
    b_acc: accuracy at best accuracy
    b_prec: accuracy at best precision
    auc: area under the roc curve score
    """

    # placeholder for aggregated predictions
    p_agg_prec, p_agg_acc = [], []

    # aggregated true labels and boundary error scores
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([(((vote_scores[s] - min(vote_scores[s])) / (max(vote_scores[s]) - min(vote_scores[s]) + 1))+0.00001).tolist() for s in vote_scores], [])

    # determine threshold based on scores and labels
    thresh_prec = get_thresh(s_agg, t_agg, maximize='prec')
    thresh_acc = get_thresh(s_agg, t_agg, maximize='acc')

    # predictions based on scores and threshold
    for scores in vote_scores.values():
        scores = (scores - min(scores) / (max(scores) - min(scores) + 1)) + 0.00001
        p_prec = scores > thresh_prec
        p_acc = scores > thresh_acc

        p_agg_prec.extend(p_prec)
        p_agg_acc.extend(p_acc)

    # evaluation metrics: best_acc, best_prec, auc_roc
    b_acc = best_acc(s_agg, t_agg)
    b_prec = get_stats(p_agg_prec, t_agg)[0]
    auc = roc_auc_score(t_agg, s_agg)
    # confusion matrix under best accuracy threshold
    conf_mat = conf_matrix(p_agg_acc, t_agg)

    if incl_cnt:
        return b_acc, b_prec, auc, len(p_agg_prec), *conf_mat
    else:
        return b_acc, b_prec, auc, len(p_agg_prec)


# evaluate prediction accuracies based on true labels and vote scores
def eval_vote_preds(eval_trues, vote_scores, incl_cnt=False):
    """ Evaluate prediction accuracies for given true labels and scores

    Parameters
    ----------
    eval_trues: dictionary of true labels for each game
    vote_scores: dictionary of final scores for voting
    incl_cnt: include absolute counts

    Returns
    ----------
    b_acc: accuracy at best accuracy
    b_prec: accuracy at best precision
    auc: area under the roc curve score
    """

    # placeholder for aggregated predictions
    p_agg_prec, p_agg_acc = [], []

    # aggregated true labels and boundary error scores
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([vote_scores[s].tolist() for s in vote_scores], [])

    # determine threshold based on scores and labels
    thresh_prec = get_thresh(s_agg, t_agg, maximize='prec')
    thresh_acc = get_thresh(s_agg, t_agg, maximize='acc')

    # predictions based on scores and threshold
    for scores in vote_scores.values():
        p_prec = scores > thresh_prec
        p_acc = scores > thresh_acc

        p_agg_prec.extend(p_prec)
        p_agg_acc.extend(p_acc)
    print(f"thresh acc - {thresh_acc}")
    print(f"thresh prec - {thresh_prec}")
    # evaluation metrics: best_acc, best_prec, auc_roc
    b_acc = best_acc(s_agg, t_agg)
    b_prec = get_stats(p_agg_prec, t_agg)[0]
    auc = roc_auc_score(t_agg, s_agg)
    # confusion matrix under best accuracy threshold
    conf_mat = conf_matrix(p_agg_acc, t_agg)

    if incl_cnt:
        return b_acc, b_prec, auc, len(p_agg_prec), *conf_mat, s_agg, t_agg, thresh_prec, thresh_acc
    else:
        return b_acc, b_prec, auc, len(p_agg_prec)

