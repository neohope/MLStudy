#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
apriori对投票结果进行关联分析
"""

from time import sleep
from Unsupervised.CorrelationAnalysis.Apriori import apriori_utils
# 这个库不支持Python3.6，需要修改两类地方
# 一个是dict相关，一个是urllib相关
from votesmart import votesmart


def getActionIds():
    """
    根据文本文件数据记录的id，调用网站API，获取数据
    """
    # 注册地址 https://votesmart.org
    votesmart.apikey = 'votesmart api key string'
    actionIdList = []
    billTitleList = []
    with open('../../../Data/Apriori/recent20bills.txt') as fr:
        for line in fr.readlines():
            billNum = int(line.split('\t')[0])
            try:
                billDetail = votesmart.votes.getBill(billNum)
                for action in billDetail.actions:
                    if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                        actionId = int(action.actionId)
                        print ('bill: %d has actionId: %d' % (billNum, actionId))
                        actionIdList.append(actionId)
                        billTitleList.append(line.strip().split('\t')[1])
            except Exception as e:
                print ("problem getting bill %d" % billNum)
                raise(e)
            sleep(1)
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):
    """
    根据文本文件数据记录的id，调用网站API，获取数据
    """
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        sleep(1)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning


if __name__ == "__main__":
    # 根据美国国会投票记录，建立数据集
    actionIdList, billTitleList = getActionIds()
    transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    dataSet = [transDict[key] for key in transDict.keys()]

    # 进行关联分析
    L, supportData = apriori_utils.apriori(dataSet, minSupport=0.3)

    # 生成规则
    rules = apriori_utils.generateRules(L, supportData, minConf=0.95)
    print (rules)