import numpy as np
import pandas as pd

class KMeans(object):

    def __init__(self, k, start_var, end_var, num_observations, data):

        np.random.seed(1)
        self.k = k
        self.start_var = start_var
        self.end_var = end_var
        self.num_observations = num_observations
        self.columns = [i for i in data.columns[start_var:end_var]]
        self.centroids = pd.DataFrame(columns=self.columns)


    def initialize_centroids(self, data):

        index = np.random.randint(low=0, high=self.num_observations)
        point = data.iloc[index, self.start_var:self.end_var]
        self.centroids = self.centroids.append(point, ignore_index=True)
        sliced_data = data.iloc[:, self.start_var:self.end_var]

        for i in range(1, self.k):

            distances = pd.DataFrame()
            
            for j in range(len(self.centroids)):
                dists = []
                each_centroid = self.centroids[self.centroids.index == j]
                for k in range(len(data.transpose().columns)):
                    each_data = data[data.index == k]
                    dists.append(self.get_euclidean_distance(each_centroid.iloc[0,:], each_data.iloc[0,:]))
                distances[j] = dists
            #print(distances)
            distances = distances.transpose()
            
            temp_list = distances.min()
            temp_list = pd.Series(temp_list)
            
            index = temp_list.idxmax()
            
            point = data.iloc[index, self.start_var:self.end_var]
            self.centroids = self.centroids.append(point, ignore_index=True)

        return self.centroids

    def get_euclidean_distance(self, point1, point2):
        
        if isinstance(point1, pd.DataFrame):
            point1 = point1.transpose()
            ans = []
            for i in point1:
                #print(point1[i])
                temp = np.square(point1[i] - point2)
                ans.append(np.sqrt(temp.sum()))
            ans = pd.Series(ans)
            
        elif isinstance(point2, pd.DataFrame):
            point2 = point2.transpose()
            ans = []
            for i in point2:
                temp = np.square(point2[i] - point1)
                ans.append(np.sqrt(temp.sum()))
            ans = pd.Series(ans)
            
        else:
            temp = np.square(point1 - point2)
            ans = np.sqrt(temp.sum())
        return ans
        
    def group_observations(self, data):

        distances = pd.DataFrame()
        sliced_data = data.iloc[:, self.start_var:self.end_var]
        
        for i in range(self.k):
            
            for j in range(len(self.centroids)):
                dists = []
                each_centroid = self.centroids[self.centroids.index == j]
                for k in range(len(sliced_data.transpose().columns)):
                    each_data = sliced_data[sliced_data.index == k]
                    dists.append(self.get_euclidean_distance(each_centroid.iloc[0,:], each_data.iloc[0,:]))
                distances[j] = dists

        distances = distances.transpose()
        groups = distances.idxmin()
        
        return groups.astype('int32')

    def adjust_centroids(self, data, groups):

        grouped_data = pd.concat([data, groups.rename('group')], axis=1)

        centroids = grouped_data.groupby('group').mean()
        return centroids

    def train(self, data, iters):

        cur_groups = pd.Series(-1, index=[i for i in range(self.num_observations)])
        i = 0
        flag_groups = False
        flag_centroids = False
        
        while i < iters and not flag_groups and not flag_centroids:

            groups = self.group_observations(data)
            
            centroids = self.adjust_centroids(data, groups)
            
            flag_groups = groups.equals(cur_groups)
            
            flag_centroids = centroids.equals(self.centroids)
                
            cur_groups = groups
            self.centroids = centroids

            i += 1
            print('Iteration', i)

        print('Done clustering!')
        return cur_groups
