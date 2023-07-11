#!/usr/bin/python3
#coding=utf-8

import pymysql
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 显示red折线图list[list[str]]
# def showRedLine(data: list):
def welfareLotteryAvgLine(dates: list, data: list):
    # 画布
    plt.figure(figsize=(100,5))
    plt.plot(dates, [int(i) for i in data], label = 'red svg')
    # plt.rcParams['font.family']='MicroSoft YaHei'  #设置字体，默认字体显示不了中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    print(plt.rcParams)
    # 设置图表标题
    plt.title('svg line')
    # x轴标题
    plt.xlabel('code')
    # y轴标题
    plt.ylabel('svg')
    # y轴范围
    plt.ylim(1, 33)
    # 显示
    plt.show()


# 链接mysql
connect = pymysql.connect(
    host='127.0.0.1',
    # host='110.41.145.30',
    db="mt_scrapy",
    user="root",
    # user='matiastang',
    passwd="MySQL_18380449615",
    charset='utf8',
    use_unicode=True,
    cursorclass=pymysql.cursors.DictCursor
)
# 通过cursor执行增删查改
cursor = connect.cursor()

# 查询
sql = """
    SELECT * from welfare_lottery_double ORDER BY code DESC LIMIT 20;
"""

codes = []
dates = []
reds = []
blues = []

try:
    cursor.execute(sql)
    #这是获取表中第一个数据
    # result = cursor.fetchone()
    #这是查询表中所有的数据
    result=cursor.fetchall()
    # code数据
    codes = [item['code'] for item in result]
    # date数据
    dates = [item['date'] for item in result]
    # red数据
    reds = [item['red'] for item in result]
    # blue数据
    blues = [item['blue'] for item in result]
    
    
except:
    print('查询失败----')
    connect.rollback()

def splitReds(reds: str) -> list[int]:
    redList = reds.split(',')
    return list([int(item) for item in redList])
    # return list(map(lambda item: int(item), redList))
# 显示
# showLine(blues[:20])
# showPie(blues, range(1, 17))
# red数据降维
allReds = np.array([v.split(',') for v in reds]).ravel()
# showRedLine([v.split(',') for v in reds])
redData = [splitReds(item) for item in reds]
print(redData)
redDatas = [list(map(lambda item: item[v], redData)) for v in range(0, 5)]

datas = list(map(lambda item: sum(item)/len(item), redData))
welfareLotteryAvgLine(codes[:10], datas[:10])
# ndarray转list
# allReds = allReds.tolist()
# showPie(allReds, range(1, 34))

# 退出
connect.close()
cursor.close()