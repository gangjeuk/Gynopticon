from abc import *
import random
from typing import *
import pandas as pd


class Server:
    # Custom 0 - weight custom
    # give more weight on history weight
    def __init__(                   
        self, user_num, cons_weight=0.05, his_weight=0.01, max_his_record=10
    ):
        self.user_num = user_num
        # higher is better
        self.validity = {f"user_{i}": 1.0 for i in range(user_num)} 
        # Lower is better
        self.dubious = {f"user_{i}": 0.0 for i in range(user_num)} 
        self.history = {f"user_{i}": [] for i in range(user_num)} 
        # C1: consensus weight
        # C2: history weight
        self.cons_weight = cons_weight
        self.his_weight = his_weight
        self.max_his_record = max_his_record

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

    def make_cons(self, votes: List[Literal[0, True, False]], target_user):
        
        # Custom 1 - real time battle. Considerng target_user only
        
        t_cnt, f_cnt = votes.count(True), votes.count(False)
        
        ground_truth = True if t_cnt > f_cnt else False
        
        for i in range(self.user_num):
            user_name = f"user_{i}"
            
            # Custom 2 - ignore self-voting
            # ignore non-paticipated user
            if i == target_user or votes[i] == None:
                continue
            
            user_lied = True if votes[i] is not ground_truth else False
        
            his_record = self.check_history(user_name)
            
            self.validity[user_name] += (self.cons_weight * (-1 if user_lied is True else 1)) \
                                    + (self.his_weight * his_record)
                                    
            self.__check_validity(user_name)
            
            if len(self.history.get(user_name)) == self.max_his_record:
                self.history.get(user_name).pop()
            self.history.get(user_name).append(False if user_lied else True)
        
        if t_cnt > f_cnt:
            s = 0.0
            # Custom 3 - reflect ratio of voters
            C = 1
            for i in range(self.user_num):
                if votes[i] is False:
                    C =  -1 * (f_cnt / self.user_num) 
                elif votes[i] is True:
                    C = (t_cnt/ self.user_num)
                elif votes[i] is None:
                    continue
                user_name = f"user_{i}"
                s += self.cons_weight * C * self.validity.get(user_name)
            
            target_user_name = f"user_{target_user}"
            self.dubious[target_user_name] += s

         

    def check_history(self, user_name):
        his = self.history.get(user_name)
        t_cnt, f_cnt = his.count(True), his.count(False)
        return t_cnt - f_cnt
    
    
    def dump(self):
        print("-----------vote-----------")
        print(pd.DataFrame(self.votes, index=[0]))
        print("----------dubious---------")
        print(pd.DataFrame(self.dubious, index=[0]))
        print("---------validity---------")
        print(pd.DataFrame(self.validity, index=[0]))
        print("---------history---------")
        print(pd.DataFrame(self.history))


