from numpy import *
import matplotlib.pyplot as plt
#根据计算结果进行分类，分类函数
def classifyVector(intX,weights):
	prob = sigmoid(sum(intX*weights))
	if prob > 0.5:return 1.0
	else:return 0.0

#进行测试	
def colicTest():
	frTrain = open('horseColicTraining.txt')#读取训练数据
	frTest = open('horseColicTest.txt')#读取测试数据
	traingSet = []; trainingLabels=[]#存储训练数据矩阵，存储训练数据分类标签
	for line in frTrain.readlines():#对读取的训练数据进行切割，并放入训练数据矩阵中
		currLine = line.strip().split('\t')
		lineArr = []
		for i in list(range(21)):#读取21个参数
			lineArr.append(float(currLine[i]))
		traingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))#读取位置22的标签
		
	traingWights = stocGradAscent1(array(traingSet),trainingLabels,1000)#对数据进行训练，次数500次
	errorCount = 0;numTestVec = 0.0#初始化错误数目和测试数目
	for line in frTest.readlines():#计算测试数目，并进行测试
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in list(range(21)):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr),traingWights))!= int(currLine[21]):#测试分类和实际进行对比
			errorCount += 1.0
	errorRate = (float(errorCount)/numTestVec)#出错数目/测试数目，计算出错率
	print("the error rate of this test is: %f"%errorRate)
	return errorRate
	
#随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	m,n = shape(dataMatrix)#计算数据矩阵的行数和列数，行数代表一组个例，列数代表参数
	weights = ones(n)#根据参数的个数设置参数的权值
	recordWeights = [];recordNumIndex =[]
	for j in list(range(numIter)):
		recordWeights.append(sum(weights)/len(weights))#对权值求平均值，来表示多维数据
		recordNumIndex.append(j)#此时权值对应的迭代次数
		dataIndex = list(range(m))
		for i in list(range(m)):
			alpha = 4/(1.0+j+i) + 0.001#随机设置学习算子，可以看出算子前期幅度较大，随后变小
			randIndex = int(random.uniform(0,len(dataIndex)))#随机选取学习个例
			h = sigmoid(sum(dataMatrix[randIndex]*weights))#进行预测
			error = classLabels[randIndex] - h#计算误差
			weights = weights + alpha*error*dataMatrix[randIndex]#调整参数权值
			del(dataIndex[randIndex])#删除学习后的个例
		
		drawPlot(recordNumIndex,recordWeights)#绘制权值的变化图像
	return weights#返回学习后的参数权值

#绘制图像权值收敛，通过对权值求平均值
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
def drawPlot(x,y):
	ax.plot(list(x),list(y))
	plt.xlabel('numIter')
	plt.ylabel('weights')
	#plt.show()
	#plt.savefig('weight.jpg')  # 保存损失当前图
	plt.pause(0.001)
	#plt.close(fig)
	
#对测试进行多次，取平均值	
def multiTest():
	numTests = 10;errorSum = 0.0
	for i in list(range(numTests)):
		errorSum += colicTest2()
	print("after %d iterations the average error rate is :%f"%(numTests,errorSum/float(numTests)))
	

#不绘制图像，用于10次错误率的
def colicTest2():
	frTrain = open('horseColicTraining.txt')#读取训练数据
	frTest = open('horseColicTest.txt')#读取测试数据
	traingSet = []; trainingLabels=[]#存储训练数据矩阵，存储训练数据分类标签
	for line in frTrain.readlines():#对读取的训练数据进行切割，并放入训练数据矩阵中
		currLine = line.strip().split('\t')
		lineArr = []
		for i in list(range(21)):#读取21个参数
			lineArr.append(float(currLine[i]))
		traingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))#读取位置22的标签
		
	traingWights = stocGradAscent2(array(traingSet),trainingLabels,500)#对数据进行训练，次数500次
	errorCount = 0;numTestVec = 0.0#初始化错误数目和测试数目
	for line in frTest.readlines():#计算测试数目，并进行测试
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in list(range(21)):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr),traingWights))!= int(currLine[21]):#测试分类和实际进行对比
			errorCount += 1.0
	errorRate = (float(errorCount)/numTestVec)#出错数目/测试数目，计算出错率
	print("the error rate of this test is: %f"%errorRate)
	return errorRate
#不绘制图像，用于10次错误率的
def stocGradAscent2(dataMatrix,classLabels,numIter=150):
	m,n = shape(dataMatrix)#计算数据矩阵的行数和列数，行数代表一组个例，列数代表参数
	weights = ones(n)#根据参数的个数设置参数的权值
	for j in list(range(numIter)):
		dataIndex = list(range(m))
		for i in list(range(m)):
			alpha = 4/(1.0+j+i) + 0.01#随机设置学习算子，可以看出算子前期幅度较大，随后变小
			randIndex = int(random.uniform(0,len(dataIndex)))#随机选取学习个例
			h = sigmoid(sum(dataMatrix[randIndex]*weights))#进行预测
			error = classLabels[randIndex] - h#计算误差
			weights = weights + alpha*error*dataMatrix[randIndex]#调整参数权值
			del(dataIndex[randIndex])#删除学习后的个例
	return weights#返回学习后的参数权值
		
#定义分类函数
def sigmoid(intX):
	return 1.0/(1+exp(-intX))