from Levenshtein import *
import numpy as np
import scipy as sp


def function(a, s):
	alpha=1
	if(s>a):
		return alpha*(s-a)
	else:
		return s-a

def CalculateS(S):
    s=[]
    for a in range(0,S.shape[0]):
    	x,y=0.0,1.0
    	while y-x>0.000001:
    		c=0.0
    		for b in range(0,S.shape[0]):
    			if a==b:
    				continue
    			c+=function((x+y)/2,S[a,b])
    		if c>0.0:
    			x=(x+y)/2
    		else:
    			y=(x+y)/2
    	s.append((x+y)/2)
    return s

def GetFunctionMatrix(s, S):
	ret=np.zeros(S.shape)
	for i in range(0,S.shape[0]):
		for j in range(0,S.shape[0]):
			ret[i,j]=function(s[i],S[i,j])
	return ret

def GetFunctionNormMatrix(s):
	ret=np.zeros((len(s),1))
	for i in range(0,len(s)):
		ret[i]=function(s[i],1)
	ret=np.tile(ret,(1,len(s)))
	return ret

def GetSumMatrix(S):
	ret=S.sum(axis=1)
	ret=np.tile(ret,(S.shape[0],1)).transpose()
	return ret

def LoadFile(fileName):
	openFile=open(fileName,encoding='utf-8')
	names={}
	Name=[]
	Label=[]
	n=0	
	lines=openFile.readlines()
	for line in lines:
		items=line.strip().split(' ')
		for i in range(0,int(len(items)/2)):
			if items[i*2] not in names:
				names[items[i*2]]=n
				n+=1
				Name.append(items[i*2])
				Label.append(int(items[i*2+1]))
	openFile.close()

	Weight=np.zeros((len(names),len(names)))
	openFile=open(fileName,encoding='utf-8')
	lines=openFile.readlines()
	for line in lines:
		items=line.strip().split(' ')
		for i in range(0,int(len(items)/2)):
			for j in  range(0,int(len(items)/2)):
				if i==j:
					continue
				Weight[names[items[i*2]],names[items[j*2]]]+=1
	openFile.close()
	#print(Weight)

	for i in range(0,len(Weight)):
		if Weight[i].sum()>0:
			Weight[i]=np.divide(Weight[i],Weight[i].sum())
		else:
			Weight[i]=np.divide(np.ones(len(names)),len(names))
	Weight=Weight.transpose()
	

	return Name,Weight,Label


def Initialize(Name, Weight, Label):
	shape=Weight.shape
	S=np.zeros(shape)
	for i in range(0,shape[0]):
		for j in range(0,shape[1]):
			if i==j:
				S[i,j]=1.0
			elif Label[i]==1 and Label[j]==1:
				S[i,j]=0.0
			else:
				# 编辑距离，共现
				S[i,j]=(1-distance(Name[i],Name[j])/max(len(Name[i]),len(Name[j])))#*((1-max(Weight[i,j],Weight[j,i]))**3)
				if S[i,j]>0.857:
					S[i,j]=1.0
	return S

def Iterate(Name, Weight, S, S0, Label):
	# 结构相似
	c1,c2=0.2,0.1
	ret1=S
	ret2=np.dot(np.dot(Weight.transpose(),S),Weight)

	# 结构互斥
	ret3=np.zeros(S.shape)
	s=CalculateS(S)
	function_matrix=GetFunctionMatrix(s,S)
	function_norm_matrix=GetFunctionNormMatrix(s)
	sum_matrix=GetSumMatrix(S)
	ret3=np.dot(S,function_matrix.transpose())/(function_norm_matrix.transpose()*sum_matrix)
	ret3+=np.dot(S,function_matrix.transpose()).transpose()/(function_norm_matrix*sum_matrix.transpose())
	'''for i in range(0,S.shape[0]):
		for j in range(0,S.shape[0]):
			sap,sbp=0.0,0.0
			for k in range(0,S.shape[0]):
				if k==k:#k!=i and k!=j:
					sap+=S[j,k]
					sbp+=S[i,k]
			for k in range(0,S.shape[0]):
				if k==k:#!=i and k!=j:
					ret3[i,j]+=(S[j,k]*function(s[i],S[i,k]))/(function(s[i],1)*sap)
					ret3[i,j]+=(S[i,k]*function(s[j],S[j,k]))/(function(s[j],1)*sbp)'''
	ret3=(ret3+1)/2
	
	#print(ret1)
	#print(ret2)	
	#print(ret3)		
	ret=(1-c1-2*c2)*ret1+c1*ret2+c2*ret3
	#print(ret)
	return ret


def Merge(S, Label):
	for i,n1 in enumerate(S):
		if Label[i]==1:
			continue
		neighbor=[(j,n2) for j,n2 in enumerate(n1) if Label[j]==1]
		neighbor=sorted(neighbor,key=lambda x:x[1],reverse=True)
		S[i,neighbor[0][0]],S[neighbor[0][0]][i]=1.0,1.0
	return S


def PrintAns(Name, S, Label, outFileName):
	openFile=open(outFileName,'w',encoding='utf-8')
	for i,n1 in enumerate(S):
		#print(n1)
		if Label[i]==1:
			openFile.write(Name[i]+' '+Name[i]+'\n')
		else:
			neighbor=[(j,n2) for j,n2 in enumerate(n1) if Label[j]==1]
			neighbor=sorted(neighbor,key=lambda x:x[1],reverse=True)
			openFile.write(Name[i]+' '+Name[neighbor[0][0]]+'\n')
	openFile.close()

def CalculateAcc(outFileName):
	standard=open('ans(stas).txt',encoding='utf-8')
	infered=open(outFileName,encoding='utf-8')

	stdDic={}
	hit,total=0.0,0.0

	lines=standard.readlines()
	for line in lines:
		items=line.strip().split(' ')
		stdDic[items[0]]=items[1]
		#total+=1

	lines=infered.readlines()
	for line in lines:
		items=line.strip().split(' ')
		if items[0] not in stdDic:
			continue
		if items[1]==stdDic[items[0]]:
			hit+=1
		else:
			pass#print(items[1]+'\t'+stdDic[items[0]])
		total+=1

	print(hit)
	print(total)
	print(hit/total)


def FindBadCase(fileName1, fileName2):
	file1=open(fileName1,encoding='utf-8')
	file2=open(fileName2,encoding='utf-8')
	standard=open('ans(stas).txt',encoding='utf-8')

	stdDic={}
	lines=standard.readlines()
	for line in lines:
		items=line.strip().split(' ')
		stdDic[items[0]]=items[1]

	infered1={}
	lines=file1.readlines()
	for line in lines:
		items=line.strip().split(' ')
		infered1[items[0]]=items[1]

	infered2={}
	lines=file2.readlines()
	for line in lines:
		items=line.strip().split(' ')
		infered2[items[0]]=items[1]


	for key in infered1:
		if infered1[key]==infered2[key]:
			continue
		if key in stdDic:
			if infered1[key]==stdDic[key]:
				print(key,infered1[key],infered2[key],1)
			elif infered2[key]==stdDic[key]:
				print(key,infered1[key],infered2[key],2)
			else:
				pass


def ConstructToyGraph():
	Name=['Abb','Abb*','bB','bcC']
	'''
	0	4	1	3
	4	0	1	2
	1	1	0	0
	3	2	0	0

	0.00	0.57 	0.50	0.60
	0.50	0.00	0.50	0.40
	0.13	0.14	0.00	0.00
	0.37	0.28	0.00	0.00
	'''
	Weight=np.array([[0.00,0.57,0.50,0.60],[0.50,0.00,0.50,0.40],[0.13,0.14,0.00,0.00],[0.37,0.28,0.00,0.00]])
	Label=[1,0,1,1]
	#print(Weight[0].sum(),Weight[:,0].sum())
	return Name,Weight,Label



def main(fileName, maxIteration, outFileName):
	Name,Weight,Label=LoadFile(fileName)
	Name,Weight,Label=ConstructToyGraph()
	S=Initialize(Name,Weight,Label)
	S0=S.copy()

	openFile=open('Weight.txt','w')	
	for each in Weight:
		openFile.write(str(each)+'\n\n')
	openFile.close()

	openFile=open('S.txt','w')

	for i in range(0,maxIteration):
		openFile.write(str(S)+'\n\n')
		S=Iterate(Name,Weight,S,S0,Label)
		#S=Merge(S,Label)
	openFile.close()
	PrintAns(Name,S,Label,outFileName)


if __name__=='__main__':
	main('data.txt',0,'ans0.txt')
	CalculateAcc('ans0.txt')

	main('data.txt',10,'ans10.txt')
	CalculateAcc('ans10.txt')

	#FindBadCase('ans0.txt','ans10.txt')




