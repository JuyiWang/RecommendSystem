import math
import numpy as np
from CBRecommend import PrepareFeature

class ItemCfRecommend():
    def __init__(self, pf):
        pf.prepare_feat()
        # self.user_feat = pf.users_feat
        self.user_dict = pf.users_dict
        self.item_feat = pf.items_feat
        self.item_num = len(self.item_feat)
    
    def ItemSmiliarity(self):
        self.item_num = len(self.item_feat)
        # 统计每个物品出现次数
        item_user_count = dict()
        # 共现矩阵
        count = np.empty((self.item_num,self.item_num))
        for user in self.user_dict.keys():
            for item in self.user_dict[user].keys():
                item_user_count[item] = item_user_count.get(item,0)+1
                for another in self.user_dict[user].keys():
                    if another != item:
                        count[int(item)-1][int(another)-1] += 1
        
        for i in range(self.item_num):
            for j in range(self.item_num):
                sq = math.sqrt(item_user_count.get(i+1,1)) * math.sqrt(item_user_count.get(j+1,1))
                count[i][j] = count[i][j]/sq 
        self.count = count

    # 简单版,所有商品统一排序
    def get_recommend_movie(self, user):
        item_score_dict = self.user_dict[user]
        score = np.empty((self.item_num))
        for key in item_score_dict.keys():
            score[int(key)-1] = int(item_score_dict[key])
        recommend_list = np.matmul(self.count, score)
        recommend = sorted([(idx+1,score) for idx,score in enumerate(recommend_list)], key = lambda k:k[1], reverse = True)
        return recommend
    
    # 复杂版,与原书一致,减小了计算量
    def com_get_recommend_movie(self, user, k = 10):
        item_score_dict = self.user_dict[user]
        recommend_dict = dict()
        for key in item_score_dict.keys():
            pi = int(item_score_dict[key])
            sim_item = [(idx,score) for idx,score in enumerate(self.count[int(key)-1])]
            topk_sim_item = sorted(sim_item, key = lambda k:k[1], reverse = True)[:k]
            for idx,weight in topk_sim_item:
                recommend_dict[idx+1] = recommend_dict.get(idx+1,0) + pi*weight
        recommend = sorted(recommend_dict.items(), key = lambda k:k[1], reverse = True)
        return recommend

    def get_movie(self, user ,nitems = 10,index = 0):
        re = self.get_recommend_movie(user) if index == 0 else self.com_get_recommend_movie(user)
        item_list = self.user_dict[user].keys()
        re_out = dict()
        for (item,score) in re:
            if item not in item_list:
                re_out[item] = score
                if len(re_out) == nitems: break
        return re_out
                

if __name__ == '__main__':
    pf = PrepareFeature()
    i_cf = ItemCfRecommend(pf)
    i_cf.ItemSmiliarity()
    print(i_cf.get_movie(1,index = 0))
    print(i_cf.get_movie(1,index = 1))
    """
    偷懒版:
    {423: 422.14133557269486, 568: 396.7741119250328, 385: 396.4752023058724, 403: 393.21218041447423, 655: 390.8157867833303}
    原书版:
    {423: 62.57288956150498, 385: 57.11226061189296, 403: 54.46852845446937, 405: 51.0243304405835, 550: 37.95412668254058}
    差距还是挺大的
    """
