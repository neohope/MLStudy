#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
预测乐高价格
"""

import json
import codecs
import urllib.request
from time import sleep
import numpy as np
from Supervised.Regression.LinearRegression import linear_regression_utils
from bs4 import BeautifulSoup


def test():
    """
    预测乐高价格
    """
    lgX = []
    lgY = []
    data_collect_offline(lgX, lgY)
    #data_collect_online(lgX, lgY)
    crossValidation(lgX, lgY, 10)


def data_collect_offline(retX, retY):
    """
    读取乐高价格
    """
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, '../../../Data/LinearRegression/lego/lego10196.html', 2009, 3263, 249.99)


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    从页面读取数据
    """
    with codecs.open(inFile, encoding='utf-8') as fi:
        soup = BeautifulSoup(fi.read(),features="html.parser")

    # 根据HTML页面结构进行解析
    i = 1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')

        if len(soldUnicde)==0:
            print ("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)


def data_collect_online(retX, retY):
    """
    google api 查询6种lego玩具价格
    """
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    sleep(10)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    sleep(10)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    sleep(10)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    sleep(10)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    sleep(10)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    """
    google api 查询lego价格
    解析Json生成数据
    fix me
    """
    myAPIstr = 'google api key string'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())

    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0

            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print ('problem with item %d' % i)


def crossValidation(xArr,yArr,numVal=10):
    """
    预测乐高价格
    """
    num_test_points=30
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal,num_test_points))

    for i in range(numVal):
        trainX=[]; trainY=[]; testX = []; testY = []
        np.random.shuffle(indexList)

        # 90%数据作为训练数据，10%数据作为测试数据
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        # 岭回归
        wMat = linear_regression_utils.ridgeTest(trainX, trainY)    #get 30 weight vectors from ridge
        for k in range(num_test_points):
            matTestX = np.mat(testX)
            matTrainX=np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)#test ridge results and store
            errorMat[i,k]=linear_regression_utils.rssError(yEst.T.A, np.array(testY))

    # 在ridge weight系数下结果分析
    meanErrors = np.mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors==minMean)]
    # regularized
    # Xreg = (x-meanX)/var(x)
    # unregularize
    # x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights/varX
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))


if __name__ == '__main__':
    test()
