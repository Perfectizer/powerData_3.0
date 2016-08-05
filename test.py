# coding:utf-8
import MySQLdb
import numpy as np
import time
import math


class DBConnect:
    def __init__(self):
        self.db = MySQLdb.connect("192.168.1.58", "dbuser", "lfmysql", "powerloaddata")

    def fetch_data(self, query_sql):
        cursor = self.db.cursor()
        cursor.execute(query_sql)
        results = cursor.fetchall()
        return results

    def close_conn(self):
        self.db.close()


class ProcessData:
    def __init__(self):
        pass

    # normalization data using (x-min)/(max-min)
    def normalize_data_min_min_max(self, data):
        min_value = min(data)
        max_value = max(data)
        return [(float(i) - min_value) / float(max_value - min_value) for i in data]

    # normalization data using (x-mean)/(max-min)
    def normalize_data_mean_min_max(self, data):
        mean_value = np.mean(data)
        min_value = min(data)
        max_value = max(data)
        return [(float(i) - mean_value) / (max_value - min_value) for i in data]

    def normalize_data_z_score(self, data):
        mean_value = np.mean(data)
        s2 = (sum([(i - mean_value) * (i - mean_value) for i in data]) / len(data)) ** 0.5
        return [(i - mean_value) / s2 for i in data]

    def cluster_data(self, data):
        pass


class Cluster:
    def __init__(self):
        pass

      def Distance(self, a, b):
        dis = 0.0
        d = []
        ang = []
        tr = []
        n = len(a)

        #Feature1: angle difference
        ang[0] = abs(math.acos((a[1]-a[0])/math.sqrt(1+(a[1]-a[0])**2))-math.acos((b[1]-b[0])/math.sqrt(1+(b[1]-b[0])**2)))
        ang[n-1] = abs(math.acos((a[n-2]-a[n-1])/math.sqrt(1+(a[n-2]-a[n-1])**2))-math.acos((b[n-2]-b[n-1])/math.sqrt(1+(b[n-2]-b[n-1])**2)))
        for i in xrange(1,n-2):
            anga = abs(math.acos((a[i-1]-a[i])/math.sqrt(1+(a[i-1]-a[i])**2))) + abs(math.acos((a[i+1]-a[i])/math.sqrt(1+(a[i+1]-a[i])**2)))
            angb = abs(math.acos((b[i-1]-b[i])/math.sqrt(1+(b[i-1]-b[i])**2))) + abs(math.acos((b[i+1]-b[i])/math.sqrt(1+(b[i+1]-b[i])**2)))
            ang[i] = abs(anga - angb) * math.pi /180
        max1 = max(ang)
        min1 = min(ang)
        delta1 = max1 - min1
        for i in xrange(0,n-1):
            ang[i] = (ang[i] - min1) / delta1

        #Feature2: absolute point difference
        for i in xrange(1,n-1):
            d[i] =  abs(a[i]-b[i])
        avg = sum(d)/n
        for i in xrange(0,n-1):
            d[i] -= avg
            d[i] = abs(d[i])
        max2 = max(d)
        min2 = min(d)
        delta2 = max2 - min2
        for i in xrange(0,n-1):
            d[i] = (d[i] - min2) / delta2

        #Feature3: up-or-down trend
        tra = []
        trb = []
        if a[1]>a[0]:
            tra[0] = 1
        elif a[1]==a[0]:
            tra[0] = 0
        else:
            tra[0] = -1
        if a[n-1]>a[n-2]:
            tra[n-1] = 1
        elif a[n-1]==a[n-2]:
            tra[n-1] = 0
        else:
            tra[n-1] = -1
        for i in xrange(1,n-2):
            if (a[i+1] > a[i]) and (a[i] > a[i-1]):
                tra[i] = 1
            elif (a[i+1] < a[i]) and (a[i] < a[i-1]):
                tra[i] = -1
            elif (a[i+1] == a[i]) and (a[i] == a[i-1]):
                tra[i] = 0
            elif (a[i+1] <= a[i]) and (a[i] >= a[i-1]):
                tra[i] = 2
            elif (a[i+1] >= a[i]) and (a[i] <= a[i-1]):
                tra[i] = -2
        if b[1]>b[0]:
            trb[0] = 1
        elif b[1]==b[0]:
            trb[0] = 0
        else:
            trb[0] = -1
        if b[n-1]>b[n-2]:
            trb[n-1] = 1
        elif b[n-1]==b[n-2]:
            trb[n-1] = 0
        else:
            trb[n-1] = -1
        for i in xrange(1,n-2):
            if (b[i+1] > b[i]) and (b[i] > b[i-1]):
                trb[i] = 1
            elif (b[i+1] < b[i]) and (b[i] < b[i-1]):
                trb[i] = -1
            elif (b[i+1] == b[i]) and (b[i] == b[i-1]):
                trb[i] = 0
            elif (b[i+1] <= b[i]) and (b[i] >= b[i-1]):
                trb[i] = 2
            elif (b[i+1] >= b[i]) and (b[i] <= b[i-1]):
                trb[i] = -2
        for i in xrange(0,n-1):
            tr[i] = abs(tra[i]-trb[i])
        max3 = max(tr)
        min3 = min(tr)
        delta3 = max3 - min3
        for i in xrange(0,n-1):
            tr[i] = (tr[i] - min3) / delta3

        for i in xrange(0,n-1):
            dis += math.sqrt((ang[i]*ang[i]+d[i]*d[i]+tr[i]*tr[i])/3)
        dis = dis / n
        return dis

    # 聚类中心初始化
    def initCentroids(self, dataset, k):
        numSamples, dim = dataset.shape
        centroids = np.zeros(k, dim)
        for i in range(k):
            index = int(np.random.uniform(0, numSamples))
            centroids[i, :] = dataset[index, :]
        return centroids

    def kMean(self, dataset, k):
        numSamples, dim = dataset.shape
        clusterAssment = np.mat(np.zeros(numSamples, 2))
        clusterChanged = True

        ##step1:init centroids
        centroids = self.initCentroids(dataset, k)

        for batch in xrange(30):
          while clusterChanged:
              clusterChanged = False;

              for i in xrange(numSamples):
                  minDist = 10000000.0
                  minIndex = 0

                  for j in range(k):
                      distance = self.Distance(dataset[i,96*(batch-1):96*batch-1], centroids[j,96*(batch-1):96*batch-1])
                      if distance < minDist:
                          minDist = distance
                          minIndex = j
                  # update data jth cluster

                  if clusterAssment[i, 0] != minIndex:
                      clusterChanged = True
                      clusterAssment[i, :] = minIndex, minDist * minDist
              # update cluster centroid
              for j in range(k):
                  poinsInCluster = dataset[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                  centroids[j, :] = np.mean(poinsInCluster, axis=0)

          print "cluster complete!"

        return centroids,clusterAssment

    def splitBatch(self,dataset):
        


if __name__ == '__main__':
    dbconnect = DBConnect()
    process_data = ProcessData()
    query_sql = "SELECT DISTINCT UserID FROM TemporalData"
    results = dbconnect.fetch_data(query_sql)
    user_ids = []

    for row in results:
        user_ids.append(row[0])

    file = open('noralization_result.txt', 'w')

    for user_id in user_ids:
        query_sql = "SELECT PowerValue FROM TemporalData WHERE UserID=" + str(user_id)
        results = dbconnect.fetch_data(query_sql)
        power_list = []
        for data in results:
            power_list.append(data)
        normalization_data = process_data.normalize_data_z_score(power_list)
        # file.write(str(user_id)+'\n')
        for result in normalization_data:
            file.write("%f\n" % (result))
            # file.write(str(result)+'\n')
    file.close()

    dbconnect.close_conn()
