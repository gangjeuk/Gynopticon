from abc import *
import random
from typing import *
import pandas as pd


class User(metaclass=ABCMeta):
    @abstractmethod
    def do_vote(self, user_lst):
        pass
    
    
# Only bad guys are lying
class Cheater(User):
    def __init__(
        self,
        idx,
        model_acc,
        is_lier=False,
        tactic: Literal["random", "select"] = "random",
        lie_freq=0.5,
    ):
        assert not (is_lier == True and tactic == None)
        self.name = f"cheater_{idx}"
        self.model_acc = model_acc
        self.is_lier = is_lier
        self.tactic = tactic
        self.lie_freq = lie_freq

    def do_vote(self, user_lst):
        ret = []
        # model eval
        for user in user_lst:
            if user.name.startswith("cheater"):
                val = random.choices(
                    [True, False], weights=[self.model_acc, 1 - self.model_acc]
                )
                # self defend
                if self.is_lier is True and self.tactic == "select" and self.name == user.name:
                    val = [False]
            elif user.name.startswith("user"):
                val = random.choices(
                    [True, False], weights=[1 - self.model_acc, self.model_acc]
                )
            ret.append(val[0])

        # do lie selectively
        if self.is_lier is True and self.tactic == "random":
            lies = []
            for i in ret:
                do_change = random.choices(
                    [True, False], weights=[self.lie_freq, 1 - self.lie_freq]
                )
                lies.append(i) if do_change is False else lies.append(not i)
            return {self.name: lies}

        return {self.name: ret}


class Benign(User):
    def __init__(self, idx, model_acc):
        self.name = f"user_{idx}"
        self.model_acc = model_acc

    def do_vote(self, user_lst):
        ret = []
        for user in user_lst:
            if user.name.startswith("cheater"):
                val = random.choices(
                    [True, False], weights=[self.model_acc, 1 - self.model_acc]
                )
            elif user.name.startswith("user"):
                val = random.choices(
                    [True, False], weights=[1 - self.model_acc, self.model_acc]
                )
            ret.append(val[0])
        return {self.name: ret}


class Server:
    # give more weight on history weight
    def __init__(
        self, benign_num, cheater_num, cons_weight=0.05, his_weight=0.01, max_his_record=10
    ):
        self.user_num = benign_num + cheater_num
        # higher is better
        self.validity = {f"user_{i}": 0.0 for i in range(benign_num)} | {
            f"cheater_{i}": 0.0 for i in range(cheater_num)
        }
        # Lower is better
        self.dubious = {f"user_{i}": 0.0 for i in range(benign_num)} | {
            f"cheater_{i}": 0.0 for i in range(cheater_num)
        }
        self.history = {f"user_{i}": [] for i in range(benign_num)} | {
            f"cheater_{i}": [] for i in range(cheater_num)
        }
        # C1: consensus weight
        # C2: history weight
        self.cons_weight = cons_weight
        self.his_weight = his_weight
        self.max_his_record = max_his_record
        self.votes = {}

    def recv_user(self, vote):
        self.votes.update(vote)

    def __cnt_TF(self, row):
        lst = []
        for i in self.votes.keys():
            lst.append(self.votes[i][row])
        return lst.count(True), lst.count(False)

    def __check_validity(self, user_name):
        if self.validity[user_name] < 0.0:
            self.validity[user_name] = 0.0
        elif self.validity[user_name] > 1.0:
            self.validity[user_name] = 1.0

    def __check_dubious(self, user_name):
        if self.dubious[user_name] < -1.0:
            self.dubious[user_name] = -1.0
        elif self.dubious[user_name] > 1.0:
            self.dubious[user_name] = 1.0

    def make_cons(self):
        # For each user evaluted by other users ==> target user
        for row, target_user in enumerate(self.votes.keys()):
            # count T/F for target user and set ground truth
            t_cnt, f_cnt = self.__cnt_TF(row)
            ground_truth = True if t_cnt > f_cnt else False

            # update each user's score based on the ground truth
            for user in self.votes.keys():
                # pass for different result for equal num: e.g 1 vs 1
                if max(t_cnt, f_cnt) == self.user_num // 2 and self.user_num % 2 == 0:
                    # goto next row
                    # break
                    ground_truth = None
                # user_Nth said True/Lie
                user_lied = True if ground_truth != self.votes.get(user)[row] else False

                # check user history
                his_record = self.check_history(user)

                # eval next validity first
                self.validity[user] += (
                    self.cons_weight * (-1 if user_lied is True else 1)
                ) + (self.his_weight * his_record)

                # user validity should 0.0 <= validity <= 1.0
                self.__check_validity(user)

                # append to history
                if len(self.history.get(user)) == self.max_his_record:
                    self.history.get(user).pop()
                self.history.get(user).append(False if user_lied else True)

            # After checking row, calc dubious
            # reuse cons_value
            s = 0.0
            C = 1 if ground_truth is True else -1
            for j in self.votes.keys():
                s += C * self.validity.get(j)
            # user dubious should -1.0 <= dubious <= 1.0
            self.dubious[target_user] += s
            #self.__check_dubious(target_user)

    def check_history(self, user_name):
        his = self.history.get(user_name)
        t_cnt, f_cnt = his.count(True), his.count(False)
        return t_cnt - f_cnt
    
    def simul_voting(self, user_lst):
        for user in user_lst:
            self.recv_user(user.do_vote(user_lst))
        self.make_cons()
    
    def dump(self):
        print("-----------vote-----------")
        print(pd.DataFrame(self.votes, index=[0]))
        print("----------dubious---------")
        print(pd.DataFrame(self.dubious, index=[0]))
        print("---------validity---------")
        print(pd.DataFrame(self.validity, index=[0]))
        print("---------history---------")
        print(pd.DataFrame(self.history))



