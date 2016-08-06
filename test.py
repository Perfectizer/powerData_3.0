# coding:utf-8
import MySQLdb
import numpy as np
import math


class DBConnect:
  def __init__(self):
    # self.db = MySQLdb.connect("192.168.1.58", "dbuser", "lfmysql", "powerloaddata")
    self.db = MySQLdb.connect("166.111.81.180", "root", "root2017", "test")

  def fetch_data(self, query_sql):
    cursor = self.db.cursor()
    cursor.execute(query_sql)
    results = cursor.fetchall()
    return results

  def fetch_one_data(self, query_sql):
    cursor = self.db.cursor()
    cursor.execute(query_sql)
    result = cursor.fetchone()
    return result

  def close_conn(self):
    self.db.close()


class ProcessData:
  def __init__(self):
    pass

  # normalization data using (x-min)/(max-min)
  def normalize_data_min_min_max(self, data):
    min_value = min(data)
    max_value = max(data)
    if max_value == min_value:
      return [0] * len(data)
    return [(float(i) - min_value) / float(max_value - min_value) for i in data]

  # normalization data using (x-mean)/(max-min)
  def normalize_data_mean_min_max(self, data):
    mean_value = np.mean(data)
    min_value = min(data)
    max_value = max(data)
    if max_value == min_value:
      return [0] * len(data)
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

  def curve_distance(self, a, b):
    dis = 0.0
    n = len(a)
    d = [0] * n
    ang = [0] * n
    tr = [0] * n

    # Feature1: angle difference

    ang[0] = abs(math.acos((a[1] - a[0]) / math.sqrt(1 + (a[1] - a[0]) ** 2)) - math.acos(
      (b[1] - b[0]) / math.sqrt(1 + (b[1] - b[0]) ** 2)))
    ang[n - 1] = abs(math.acos((a[n - 2] - a[n - 1]) / math.sqrt(1 + (a[n - 2] - a[n - 1]) ** 2)) - math.acos(
      (b[n - 2] - b[n - 1]) / math.sqrt(1 + (b[n - 2] - b[n - 1]) ** 2)))
    for i in xrange(1, n - 2):
      anga = abs(math.acos((a[i - 1] - a[i]) / math.sqrt(1 + (a[i - 1] - a[i]) ** 2))) + abs(
        math.acos((a[i + 1] - a[i]) / math.sqrt(1 + (a[i + 1] - a[i]) ** 2)))
      angb = abs(math.acos((b[i - 1] - b[i]) / math.sqrt(1 + (b[i - 1] - b[i]) ** 2))) + abs(
        math.acos((b[i + 1] - b[i]) / math.sqrt(1 + (b[i + 1] - b[i]) ** 2)))
      ang[i] = abs(anga - angb) * math.pi / 180
    max1 = max(ang)
    min1 = min(ang)
    delta1 = max1 - min1
    for i in xrange(0, n - 1):
      ang[i] = (ang[i] - min1) / delta1

    # Feature2: absolute point difference
    for i in xrange(1, n - 1):
      d[i] = abs(a[i] - b[i])
    avg = sum(d) / n
    for i in xrange(0, n - 1):
      d[i] -= avg
      d[i] = abs(d[i])
    max2 = max(d)
    min2 = min(d)
    delta2 = max2 - min2
    for i in xrange(0, n - 1):
      d[i] = (d[i] - min2) / delta2

    # Feature3: up-or-down trend
    tra = [0] * n
    trb = [0] * n
    if a[1] > a[0]:
      tra[0] = 1
    elif a[1] == a[0]:
      tra[0] = 0
    else:
      tra[0] = -1
    if a[n - 1] > a[n - 2]:
      tra[n - 1] = 1
    elif a[n - 1] == a[n - 2]:
      tra[n - 1] = 0
    else:
      tra[n - 1] = -1
    for i in xrange(1, n - 2):
      if (a[i + 1] > a[i]) and (a[i] > a[i - 1]):
        tra[i] = 1
      elif (a[i + 1] < a[i]) and (a[i] < a[i - 1]):
        tra[i] = -1
      elif (a[i + 1] == a[i]) and (a[i] == a[i - 1]):
        tra[i] = 0
      elif (a[i + 1] <= a[i]) and (a[i] >= a[i - 1]):
        tra[i] = 2
      elif (a[i + 1] >= a[i]) and (a[i] <= a[i - 1]):
        tra[i] = -2
    if b[1] > b[0]:
      trb.append(1)
    elif b[1] == b[0]:
      trb.append(0)
    else:
      trb.append(-1)
    if b[n - 1] > b[n - 2]:
      trb[n - 1] = 1
    elif b[n - 1] == b[n - 2]:
      trb[n - 1] = 0
    else:
      trb[n - 1] = -1
    for i in xrange(1, n - 2):
      if (b[i + 1] > b[i]) and (b[i] > b[i - 1]):
        trb[i] = 1
      elif (b[i + 1] < b[i]) and (b[i] < b[i - 1]):
        trb[i] = -1
      elif (b[i + 1] == b[i]) and (b[i] == b[i - 1]):
        trb[i] = 0
      elif (b[i + 1] <= b[i]) and (b[i] >= b[i - 1]):
        trb[i] = 2
      elif (b[i + 1] >= b[i]) and (b[i] <= b[i - 1]):
        trb[i] = -2
    for i in xrange(0, n - 1):
      tr[i] = abs(tra[i] - trb[i])
    max3 = max(tr)
    min3 = min(tr)
    delta3 = max3 - min3
    for i in xrange(0, n - 1):
      tr[i] = (tr[i] - min3) / delta3

    for i in xrange(0, n - 1):
      dis += math.sqrt((ang[i] * ang[i] + d[i] * d[i] + tr[i] * tr[i]) / 3)
    dis = dis / n
    return dis

  # 聚类中心初始化

  def initcentroids(self, dataset, k):
    numSamples, dim = dataset.shape
    centroids = np.zeros(k, dim)
    for i in range(k):
      index = int(np.random.uniform(0, numSamples))
      centroids[i, :] = dataset[index, :]
    return centroids

    # 对n个用户第i天的数据进行聚类，对每个用户的该段数据产生类标
  def kmeans(self, dataset, k):
    numSamples, dim = dataset.shape
    clusterAssment = np.mat(np.zeros(numSamples, 2))
    clusterChanged = True

    ##初始化聚类中心
    centroids = self.initcentroids(dataset, k)

    while clusterChanged:
      clusterChanged = False;

      for i in xrange(numSamples):
        minDist = 1
        minIndex = 0

        for j in range(k):
          distance = self.curve_distance(dataset[i, :], centroids[j, :])
          if distance < minDist:
            minDist = distance
            minIndex = j
        # 对第j类进行更新
        if clusterAssment[i, 0] != minIndex:
          clusterChanged = True
          clusterAssment[i, :] = minIndex, minDist * minDist
      # 更新聚类中心
      for j in range(k):
        poinsInCluster = dataset[np.nonzero(clusterAssment[:, 0].A == j)[0]]
        centroids[j, :] = np.mean(poinsInCluster, axis=0)

    # 改进部分，对于一个数据，分别计算他与每个簇中心的距离，然后计算一个数据该簇的概率

    for i in xrange(numSamples):
      membership = np.zeros(k, 1)
      distance = np.zeros(k, 1)
      count = 0.0
      for j in range(k):
        distance[j] = 1 - self.curve_distance(dataset[i, :], centroids[j, :])
        membership.append(distance[j])
        count += distance[j]
      for j in range(k):
        membership[j] = membership[j] / count
      centroidsNo = [x for x in range(0, k)]
      item = self.random_pick(centroids, membership)
      clusterAssment[i, :] = item, distance[item]
      # 计算出第i个数据对每个中心的隶属度概率

    print "cluster complete!"
    return centroids, clusterAssment

  # 随机操作,依一定概率逃逸更改类标
  def random_pick(self, some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
      cumulative_probability += item_probability
      if x < cumulative_probability: break
    return item

  # 确定统计数最大的类标
  def label(self, li):
    max = 0
    n = len(li)
    sk = 1
    for i in range(n):
      if li[i] > max:
        sk = i
    return sk

  # 分段操作,并对结果进行统计投票
  def batch_cluster(self, dataset, cycle, k):
    # 初始化
    (n, m) = np.shape(dataset)
    data = np.matrix(dataset)
    num = m // cycle + (m % cycle != 0)
    table = [[0] * 300 for i in range(n)]
    res = []
    # 对每个batch处理,但是需要对最后一个batch单独处理以防出现不能整除的情况
    for i in range(num - 1):
      batch = data[:, (i * cycle):((i + 1) * cycle - 1)]  # 提取每一个cycle的数据
      cl = self.kmeans(batch, k)  # 调用kMean函数,返回该cycle数据反馈的用户类标
      # 对每个用户的分类计数
      for j in range(n):
        table[j][cl[j]] += 1
    batch = data[:, ((num - 2) * cycle):m]  # 提取每一个cycle的数据
    cl = self.kmeans(batch, k)  # 调用kMean函数,返回该cycle数据反馈的用户类标
    # 对每个用户的分类计数
    for j in range(n):
      table[j][cl[j]] += 1
    # 统计投票,将每个用户得票最多的类标作为当前其类标真值
    for i in range(n):
      res[i] = self.label(table[i])
    return res


if __name__ == '__main__':
  dayNumber = 30
  print ("here")
  dbconnect = DBConnect()
  process_data = ProcessData()
  query_sql = "SELECT DISTINCT USERID FROM temporaldata"
  results = dbconnect.fetch_data(query_sql)
  user_ids = []

  for row in results:
    user_ids.append(row[0])

  file = open('noralization_result.txt', 'a')
  mat = []
  vocation_type = [-1] * len(user_ids)
  file.write(" UserID     PowerValue \n")
  line = 0

  for user_id in user_ids:
    query_sql = "SELECT PowerValue FROM datapowerhistory WHERE UserID=" + str(user_id) + " LIMIT 0," + str(
      96 * dayNumber)
    results = dbconnect.fetch_data(query_sql)
    if not results:
      continue
    power_list = [0] * (96 * dayNumber)

    i = 0
    for data in results:
      power_list[i] = data[0]
      i += 1

    query_sql = "SELECT devvipinfo.VOCATIONTYPE FROM devvipinfo WHERE devvipinfo.ID=" + str(user_id)
    vocation_type[line] = dbconnect.fetch_one_data(query_sql)[0]

    normalization_data = [0] * len(power_list)
    for day in range(dayNumber):
      undealData = power_list[(day * 96):((day + 1) * 96 - 1)]
      dealedData = process_data.normalize_data_min_min_max(undealData)
      normalization_data[(day * 96):((day + 1) * 96 - 1)] = dealedData
    # file.write(str(user_id)+'\n')
    # for result in normalization_data:
    #   file.write('{0:^9d} {1:<}'.format(user_id, result))
    #   file.write('\n')
      # file.write(str(result)+'\n')
    mat.append(normalization_data)
    line += 1
  mat = np.matrix(mat)
  #print np.shape(mat)
  obj = Cluster()
  ans = obj.batch_cluster(mat)

  file.close()

  dbconnect.close_conn()
