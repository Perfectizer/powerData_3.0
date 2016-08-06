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

  def close_conn(self):
    self.db.close()


if __name__ == '__main__':
  print ("here")
  dbconnect = DBConnect()
  query_sql = "SELECT DISTINCT ID FROM devvipinfo"
  results = dbconnect.fetch_data(query_sql)

  for row in results:
    print(row[0])