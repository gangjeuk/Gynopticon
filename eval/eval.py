import os, sys

sys.path.append(os.path.dirname(".."))
import matplotlib.pyplot as plt
from res.Botscreen.utils import *
import logging
from logging import debug, info
from utils import Config
e = exp_from_arguments()

if Config['debug']:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

########## Gynopticon #########
# evaluation result
res = []
table_output = pd.DataFrame()
dubious_scores = {}
########## Gynopticon-end #########

# split experiments into k folds
for i,test_games in enumerate(e.splits):

    # run on single split
    if e.split >= 0:
        if i != e.split:
            continue

    info(f'### split {i} ###' +'\n')

    train_games = sorted(set(e.all_games) - set(test_games))

    # directories for gru models
    model_dir = f'trained_models/gru_k{i}.pt'

    data_train = pd.concat(smooth_df(v) for (g,o,p),v in e.frames.items()\
                            if g in train_games and o==p and p not in e.cheater[g])

    e.norm_args = data_train[e.data_cols].min(), data_train[e.data_cols].max()

    ### configuring model ###
    info('configuring models ...\n')

    # try to load pretrained model
    try:
        saved_model = torch.load(model_dir, map_location='cpu')
        info('... saved gru model found! loading model ... ')
        model = eval(e.model)(*saved_model['args']).cpu()
        model.load_state_dict(saved_model['state'])
        model_loaded = True
    except:
        model_loaded = False

    # error if the model failed to load or in different config
    if not e.chk_config or not model_loaded:
        logging.error('error!\n')
        exit()

    info('done\n')

    # evaluation mode
    model.eval()

    ##### evaluation #####
    info('evaluation ...\n')

    ### directory for pre-evaluated results
    eval_dir = f'trained_models/eval_k{i}'

    # try to load existing eval results
    try:
        eval_trues, eval_scores = pickle.load(open(eval_dir,'rb'))
        assert set(eval_trues.keys()) == set(test_games)

    except:
        # true labels and error scores for each game
        eval_trues, eval_scores = {}, {}

        # for each game in test games
        for g in test_games:
            # get number of players from experiment info
            _,l = e.exp_info[g[0]]
            player_id = range(l)

            trues, scores = np.zeros((l,l)), np.zeros((l,l))
            for o,p in product(player_id, repeat=2):
                trues[o,p], scores[o,p] = \
                                anom_score(m=model,
                                        e=e,
                                        key=(g,str(o),str(p)),
                                        use_all=False)

            eval_trues[g] = trues
            eval_scores[g] = scores

        # save results
        pickle.dump((eval_trues, eval_scores), open(eval_dir,'wb'))

    info('done\n\n')

##################%%% Gynopticon %%%###################


    info('Evaluation result loaded\n')


    # Get thresh
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([get_median(s) for s in eval_scores.values()], [])
    thresh = get_thresh(s_agg, t_agg, maximize='prec')


    # Get time series data of abnormal score 
    
    # for each match
    for match in test_games:
        from model.customized import Server
        
        info(f"evaluate match: {match}")
        _, player_num = e.exp_info[match[0]]
        
        server = Server(user_num=player_num)

        player_id = range(player_num)
        
        cnt = dict(zip([f'user_{i}' for i in player_id], [0 for _ in player_id]))
        
        scores, times = {}, {}
        
        total_series = np.array([])
        
        rst_dir = f'trained_models/vote-without-liar-{i}-{match}.pickle'
        # oberser and observed player
        try:
            scores, times, battles = pickle.load(open(rst_dir, 'rb'))
            total_series = battles['timeline']
        except:
            for ob, obd in product(player_id, repeat=2):
                scores[(ob, obd)], times[(ob, obd)] = anom_times_all(m=model,
                                                                e=e,
                                                                key=(match,str(ob),str(obd)),
                                                                use_all=True)
                
                total_series = np.concatenate((total_series, times[(ob, obd)]))
        
            total_series = np.sort(total_series)
            total_series = np.unique(total_series)
            

            battles = {"match": match, "timeline": total_series, "cheater": e.cheater[match], \
                       "battle":[{"votes":[], "voter":set(), \
                            "dubious":[], \
                            "start":0, "end":0}] \
                      }
    
        for battle_slide in consecutive_groups(total_series):
            battle_slide = list(battle_slide)
            b_bgn, b_end = battle_slide[0], battle_slide[-1]
            votes = [[None for _ in player_id] for _ in player_id]
            target_obd = []
            do_vote = False
            voter = set()
            for ob, obd in product(player_id, repeat=2):
                obd_slide = np.where(np.logical_and(b_bgn < times[(ob,obd)], times[(ob,obd)] < b_end))[0]
                if len(obd_slide) > 1:
                    max_score = np.max(scores[(ob,obd)][obd_slide[0]:obd_slide[-1]])
                    # Custom 3 - ignore self voting
                    if max_score > thresh:
                        do_vote = True
                        votes[ob][obd] = True
                        voter |= set(str(ob))
                        target_obd.append(obd)
                        
            target_obd = list(set(target_obd))
            
            if do_vote is True:
                # All users
                for target in player_id:
                    for ob in player_id:
                        obd_slide = np.where(np.logical_and(b_bgn < times[(ob,target)], times[(ob,target)] < b_end))[0]
                        if len(obd_slide) > 1:    
                            max_score = np.max(scores[(ob,target)][obd_slide[0]:obd_slide[-1]])
                            # Custom 2 - limit voter to real observer
                            if max_score < thresh and len(obd_slide) >= 20 and target != ob:
                                votes[ob][target] = False
                                voter |= set(str(ob))     
                for target in player_id:
                    vote_to_obd = list(zip(*votes))[target]
                    t_cnt, f_cnt = vote_to_obd.count(True), vote_to_obd.count(False)
                    if (t_cnt + f_cnt) > 2:
                        if (t_cnt > f_cnt):
                            cnt[f'user_{target}'] += 1
                            #debug(f'target{target} - {t_cnt}, {f_cnt} : {vote_to_obd}')
                        server.make_cons(vote_to_obd, target)       
                '''
                # Only target user
                for target in target_obd:
                    for ob in player_id:
                        obd_slide = np.where(np.logical_and(b_bgn < times[(ob,target)], times[(ob,target)] < b_end))[0]
                        if len(obd_slide) > 1:    
                            max_score = np.max(scores[(ob,target)][obd_slide[0]:obd_slide[-1]])
                            # Custom 2 - limit voter to real observer
                            if max_score < thresh and len(obd_slide) >= 10 and target != ob:
                                votes[ob][target] = False
                                voter |= set(str(ob))
                                
                for target in target_obd:
                    vote_to_obd = list(zip(*votes))[target]
                    t_cnt, f_cnt = vote_to_obd.count(True), vote_to_obd.count(False)
                    if (t_cnt + f_cnt) > 2:
                        debug(f'target{target} - {t_cnt}, {f_cnt} : {votes}')
                        if t_cnt > f_cnt and (t_cnt + f_cnt) > 2:
                            server.make_cons(vote_to_obd, target)
                '''
            battle = {"votes": votes, "voter": voter, \
                      "start": b_bgn, "end": b_end, \
                      "dubious": server.dubious, "validity": server.validity}
            
            battles['battle'].append(battle)

        debug(f'cnt: {cnt}')
        debug(f'cheater: {e.cheater[match]}')
        debug(f'final dubious score: {server.dubious}')
        debug(f'final validity score: {server.validity}')

        # end of match
        dubious_scores[match] = np.array(list(server.dubious.values()))
        pickle.dump((scores, times, battles), open(rst_dir, 'wb'))
        table_output = table_output.append(pd.DataFrame([cnt, server.dubious, server.validity], [[match]*3,['count', 'dubious', 'validity']]))
    # end of splited games #
    res.append(eval_vote_preds(eval_trues, dubious_scores, incl_cnt=True))
    dubious_scores = {}
# aggregating results from all splits
res = np.array(res)
print(table_output.to_latex(float_format="%.2f", formatters={'count': int}, longtable=True, na_rep=0))
b_acc, b_prec, auc = np.average(res[:,:3], axis=0, weights=res[:,3])
tp, tn, fp, fn = np.sum(res[:,4:8], axis=0, dtype=int)
info(f'best_acc: {b_acc:.4f}, best_prec: {b_prec:.4f}, auc_roc: {auc:.4f}')
info(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
##################%%% Gynopticon end %%%###################
