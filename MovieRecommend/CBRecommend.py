import json
import pandas as pd
import numpy as np
import math
import random

class PrepareFeature():
    def __init__(self):
        self.root = './MovieRecommend/MovieData'
    
    def prepare_item_feat(self):
        path = self.root + '/movie_data.csv'
        items = pd.read_csv(path)
        items_ids = set(items['MovieID'].values)
        items_feat = list()
        self.items_dict = {}
        for item in items_ids:
            feat = items[items['MovieID'] == item][['Unknown','Action','Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].values[0]
            items_feat.append(feat)
            self.items_dict.setdefault(item,[]).extend(feat)
        self.items_feat = np.array(items_feat)        
        
    def prepare_user_feat(self):
        path = self.root + '/rating_data.csv'
        users = pd.read_csv(path)
        users_ids = set(users['UserID'].values)
        users_dict = {}
        for user in users_ids:
            users_dict.setdefault(user,{})
        for index in users.values:
            (user, item, score) = index[:3]
            users_dict[user][item] = int(score)
        user_matrix = []
        for user in users_dict.keys():
            score_list = list(users_dict[user].values())
            socre_avg = np.mean(np.array(score_list))
            genre_list = [0]*19
            score_list = [0]*19
            i = 0
            for item in users_dict[user]:
                genre = self.items_dict[item]
                for idx in range(19):
                    if genre[idx]:
                        genre_list[idx] += 1
                        score_list[idx] += (users_dict[user][item]-socre_avg)
            for idx in range(19):
                if genre_list[idx]:
                    score_list[idx] = score_list[idx]/genre_list[idx]
            user_matrix.append(score_list)
        self.users_dict = users_dict
        self.users_feat = np.array(user_matrix)
    
    def prepare_feat(self):
        self.prepare_item_feat()
        self.prepare_user_feat()
        print("Feature build successfully~")
        
class CBRecoomend():
    def __init__(self,K, pf):
        pf.prepare_feat()
        self.K = K 
        self.user_feat = pf.users_feat
        self.user_dict = pf.users_dict
        self.item_feat = pf.items_feat

    def get_none_score(self,user):
        check_list = set(self.user_dict[user].keys())
        user_para = np.array(self.user_feat[user])
        item_para = self.item_feat
        score_list = np.matmul(item_para, user_para)
        score = [(idx+1,score) for idx,score in enumerate(score_list) if idx+1 not in check_list]
        score = sorted(score, key = lambda k:k[1],reverse = True)
        return score
    
    def get_recommend_movie(self,user):
        socre_list = self.get_none_score(user)
        return socre_list[:self.K]


if __name__ == '__main__':
    pf = PrepareFeature()
    pf.prepare_feat()
    cb = CBRecoomend(10,pf)
    print(cb.get_recommend_movie(888))

    """
    Result:
    [(1065, 0.7417375816346723), (847, 0.6008343558282209), 
    (1366, 0.6008343558282209), (1561, 0.6008343558282209), 
    (32, 0.5644171779141103), (48, 0.5644171779141103), 
    (75, 0.5644171779141103), (115, 0.5644171779141103), 
    (119, 0.5644171779141103), (320, 0.5644171779141103)]
    """