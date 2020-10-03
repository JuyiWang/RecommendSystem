import json
import pandas as pd
import numpy as np

class DataProcessing():

    def __init__(self):
        self.load_root = './MovieRecommend/ml-100k'
        self.save_root = './MovieRecommend/MovieData'

    def process(self):
        self.process_user_data()
        self.process_rate_data()
        self.process_item_data()

    def process_user_data(self):
        load_path = self.load_root + '/u.user'
        data = pd.read_csv(load_path, names = ['UserID','Age','Gender','Occupation','Zip-Code'], sep = '|')
        save_path = self.save_root + '/user_data.csv'
        data.to_csv(save_path,index = False)

    def process_item_data(self,):
        load_path = self.load_root + '/u.item'
        data = pd.read_csv(load_path, names = ['MovieID', 'Title', 'ReleaseDate', 'VideoReleaseDate','IMDbURL' , 'Unknown', 'Action', 
            'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], sep = '|')
        save_path = self.save_root + '/movie_data.csv'
        data.to_csv(save_path,index = False)

    def process_rate_data(self):
        load_path = self.load_root + '/u.data'
        data = pd.read_csv(load_path, names = ['UserID','MovieID','Rating','TimeStamp'], sep = '\t',encoding = 'utf-8')
        print(data.head())
        save_path = self.save_root + '/rating_data.csv'
        data.to_csv(save_path,index = False)

if __name__ == '__main__':
    dp = DataProcessing()
    dp.process()