#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Frequent Pattern Growth查找频繁项集
Twitter搜索结果关联性分析
"""

import twitter
import re
from time import sleep
from Unsupervised.CorrelationAnalysis.FrequentPattenGrowth import fpg_utils


def getLotsOfTweets(searchStr):
    """
    调用Twitter API，获取搜索结果页面
    """
    CONSUMER_KEY = 'CONSUMER_KEY'
    CONSUMER_SECRET = 'CONSUMER_SECRET'
    ACCESS_TOKEN_KEY = 'ACCESS_TOKEN_KEY'
    ACCESS_TOKEN_SECRET = 'ACCESS_TOKEN_SECRET'
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, access_token_key=ACCESS_TOKEN_KEY, access_token_secret=ACCESS_TOKEN_SECRET)

    # 获取15个页面
    resultsPages = []
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(2)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    """
    获取频繁项集
    """
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))

    # 创建FP树
    initSet = fpg_utils.createInitSet(parsedList)
    myFPtree, myHeaderTab = fpg_utils.createTree(initSet, minSup)

    # 创建条件FP树
    myFreqList = []
    fpg_utils.mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)

    return myFreqList


def textParse(bigString):
    """
    解析页面内容
    """
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


if __name__ == "__main__":
    # 获取数据
    lotsOtweets = fpg_utils.getLotsOfTweets('RIMM')
    listOfTerms = mineTweets(lotsOtweets, 20)
    for t in listOfTerms:
        print(t)

