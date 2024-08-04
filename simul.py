from typing import *
import pandas as pd
from model import Server, Benign, Cheater



def simulate_without_liar(
    played_match=1,
    vote_per_match=3,
    model_acc=0.8,
    benign_num=2,
    cheater_num=1,
):
    user_lst = [Benign(i, model_acc) for i in range(benign_num)]
    user_lst += [Cheater(i, model_acc, is_lier=False) for i in range(cheater_num)]

    benign, cheater = {}, {}
    for match in range(played_match):
        # Reset Server and repeat evaluation for REPORT_CNT times
        server = Server(benign_num, cheater_num)
        # Evalute for test_cnt times
        for vote in range(vote_per_match):
            server.simul_voting(user_lst)
        for user in user_lst:
            dub = server.dubious[user.name]
            val = server.validity[user.name]
            if user.name.startswith("user"):
                benign[f"{match}-{user.name}"] = (dub, val)
            elif user.name.startswith("cheater"):
                cheater[f"{match}-{user.name}"] = (dub, val)

    return benign, cheater, server


def simulate_with_liar(
    played_match=1,
    vote_per_match=3,
    model_acc=0.8,
    benign_num=2,
    cheater_num=1,
    tactic: Literal["random", "select"] = "random",
    lie_freq = 0.5,
):
    user_lst = [Benign(i, model_acc) for i in range(benign_num)]
    user_lst += [
        Cheater(i, model_acc, is_lier=True, tactic=tactic, lie_freq=lie_freq)
        for i in range(cheater_num)
    ]

    benign, cheater = {}, {}
    for match in range(played_match):
        # Reset Server and repeat evaluation for REPORT_CNT times
        server = Server(benign_num, cheater_num)
        # Evalute for test_cnt times
        for vote in range(vote_per_match):
            server.simul_voting(user_lst)
        for user in user_lst:
            dub = server.dubious[user.name]
            val = server.validity[user.name]
            if user.name.startswith("user"):
                benign[f"{match}-{user.name}"] = (dub, val)
            elif user.name.startswith("cheater"):
                cheater[f"{match}-{user.name}"] = (dub, val)

    return benign, cheater, server
