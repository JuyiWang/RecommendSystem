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
            items_feat.extend(feat)
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
        user_matirx = []
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
            user_matirx.append(score_list)
        self.user_matrix = np.array(user_matirx)
        
class CBRecoomend():
    def __init__(self):
        pass 


if __name__ == '__main__':
    pf = PrepareFeature()
    pf.prepare_item_feat()
    pf.prepare_user_feat()