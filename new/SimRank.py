from Levenshtein import *
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

class SimRank(object):

	def __init__(self, c, iteration, graph_file, standard_file):
		self.number = 0						# 所有疾病名称个数
		self.nodes = []						# 所有的节点（名字）存入数组
		self.nodes_index = {}				# <节点名，节点编号>
		self.damp1 = c 						# 阻尼系数
		self.damp2 = c
		self.labels = []					# 所有节点标记是否标准 
		self.trans_matrix = np.matrix(0)	# 转移概率矩阵
		self.weight_matrix = np.matrix(0)	# 权重矩阵，即次数
		self.sim_matrix = np.matrix(0)		# 节点相似度矩阵
		self.init_sim_matrix = np.matrix(0)	# 只考虑编辑距离的节点相似度矩阵
		self.iteration = iteration			# 最大迭代次数
		self.link = {}						# 点的邻居集合字典
		if graph_file != '':
			self.init_param(graph_file,standard_file)

	def init_param(self, graph_file, standard_file):
		# 加载标准疾病
		std_dic={}					# 索引-标准疾病名称
		openFile=open(standard_file,encoding='utf-8')
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			std_dic[items[1]]=items[0]
		openFile.close()

		# 加载文件，构建伴病网络
		openFile=open(graph_file,encoding='utf-8')
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			for item in items:
				index, name=item.strip().split('##')
				if name not in self.nodes:
					self.nodes.append(name)
					self.nodes_index[name]=len(self.nodes)-1
					if name in std_dic:
						self.labels.append(1)
					else:
						self.labels.append(0)
		#print(sum(self.labels))
		self.number=len(self.nodes)
		self.trans_matrix=np.zeros((self.number,self.number))
		self.sim_matrix=np.zeros((self.number,self.number))
		openFile.close()

		# 初始化相似度矩阵
		for i in range(0,self.number):
			for j in range(0,self.number):
				if i==j:
					self.sim_matrix[i,j]=1.0
				elif self.labels[i]==1 and self.labels[j]==1:
					self.sim_matrix[i,j]=0.0
				else:
					self.sim_matrix[i,j]=(1.0-distance(self.nodes[i],self.nodes[j])/max(len(self.nodes[i]),len(self.nodes[j])))#*((1-max(self.trans_matrix[i,j],self.trans_matrix[j,i]))**3)
		self.init_sim_matrix=self.sim_matrix.copy()

		self.merge()

		# 初始化转移概率矩阵
		openFile=open(graph_file,encoding='utf-8')
		lines=openFile.readlines()
		for line in lines:
			names=[]
			items=line.strip().split('\t')
			for item in items:
				#print(item)
				std_index,name=item.strip().split('##')
				names.append(name)
			for name1 in names:
				for name2 in names:
					if name1==name2:
						continue
					self.trans_matrix[self.nodes_index[name1],self.nodes_index[name2]]+=1
		openFile.close()
		
		self.weight_matrix=self.trans_matrix.transpose().copy()
		for i in range(0,self.number):
			if self.trans_matrix[i].sum()>0:
				self.trans_matrix[i]=np.divide(self.trans_matrix[i],self.trans_matrix[i].sum())
			else:
				pass#self.trans_matrix[i]=np.divide(np.ones(self.number),self.number)
		self.trans_matrix=self.trans_matrix.transpose()
		self.trans_matrix=np.where(self.weight_matrix>5,self.trans_matrix,0.0)
		self.trans_matrix=np.where(self.trans_matrix<0.9,self.trans_matrix,1.0)

	# 合并结点：对初始相似度高的结点对进行合并（标准和非标准之间），为了增加结构信息
	# 是否有用？看起来准确度提高了一点
	def merge(self):
		for i in range(0,self.number):
			for j in range(0,self.number):
				if self.labels[j]==0:
					continue
				if self.sim_matrix[i,j]>0.85:
					self.nodes_index[self.nodes[i]]=j


	def iterate(self):
		#self.sim_matrix = (1 - self.damp) * self.sim_matrix + self.damp * np.dot(np.dot(self.trans_matrix.transpose(), self.sim_matrix), self.trans_matrix)
		ret1=self.sim_matrix

		# 结构相似
		ret2=np.dot(np.dot(self.trans_matrix.transpose(),self.sim_matrix),self.trans_matrix)

		'''w0,w1,w2=u'急性后壁心肌梗死',u'急性心肌梗死',u'急性再发心肌梗死'
		i0,i1,i2=self.nodes_index[w0],self.nodes_index[w1],self.nodes_index[w2]
		print(ret1[i0,i1],ret1[i0,i2])
		print(ret2[i0,i1],ret2[i0,i2])'''

		# 结构互斥
		# 改成矩阵提高计算效率
		ret3=np.zeros(self.sim_matrix.shape)
		s=self.CalculateS()
		function_matrix=self.GetFunctionMatrix(s)
		function_norm_matrix=self.GetFunctionNormMatrix(s)
		sum_matrix=self.GetSumMatrix()
		ret3=np.dot(self.sim_matrix,function_matrix.transpose())/(function_norm_matrix.transpose()*sum_matrix)
		ret3+=np.dot(self.sim_matrix,function_matrix.transpose()).transpose()/(function_norm_matrix*sum_matrix.transpose())
		ret3=(ret3+1)/2
		
		self.sim_matrix=(1-self.damp1-2*self.damp2)*ret1+self.damp1*ret2+self.damp2*ret3

		for i in range(0,self.number):
			self.sim_matrix[i,i]=1.0


	def run_sim_rank(self):
		# 记录网络信息，包括固定值（节点，转移矩阵），和变化值（相似矩阵）
		openFile=open('graph.txt','w',encoding='utf-8')
		openFile.write('nodes:'+str(self.nodes)+'\n\n')
		openFile.write('trans_matrix'+str(self.trans_matrix)+'\n\n')
		for i in range(0,self.iteration):
			openFile.write(str(i)+str(self.sim_matrix)+'\n')
			self.test_sim_rank('test.txt','test_ans/ans'+str(i)+'.txt')
			self.iterate()
		openFile.close()

	def test_sim_rank(self, test_file, ans_file):
		# 加载测试文件，计算结果
		names=[]
		hit,total=0.0,0.0
		openFile=open(test_file,encoding='utf-8')
		ansFile=open(ans_file,'w',encoding='utf-8')
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			for item in items:
				name, std_name=item.strip().split('##')
				if name in names or std_name=='NONE':
					continue
				names.append(name)
				if self.labels[self.nodes_index[name]]==1:
					hit+=1
				else:
					sim_tmp=self.sim_matrix[self.nodes_index[name]]
					sim_tmp=[(i,sim) for i,sim in enumerate(sim_tmp) if self.labels[i]==1]
					max_sim=max(sim_tmp,key=lambda x:x[1])
					if (self.nodes[max_sim[0]]==std_name):
						hit+=1
					ansFile.write(name+':'+self.nodes[max_sim[0]]+'\t'+std_name+'\n')
				total+=1
		print(hit,total,hit/total)
		openFile.close()
		ansFile.close()


	def function(self, a, s):
		alpha=1
		if(s>a):
			return alpha*(s-a)
		else:
			return s-a

	def function_syn(self, a, s):
		# 并行计算一行的function值
		alpha=1.0
		ret=s-a
		ret=np.where(ret<0,ret,ret*alpha)
		return ret

	def CalculateS(self):
		s=[]
		for a in range(0,self.number):
			x,y=0.0,1.0
			while y-x>0.001:
				c=self.function_syn((x+y)/2,self.sim_matrix[a]).sum()
				if c>0.0:
					x=(x+y)/2
				else:
					y=(x+y)/2
			s.append((x+y)/2)
		return s

	def GetFunctionMatrix(self,s):
		ret=np.zeros(self.sim_matrix.shape)
		for i in range(0,self.number):
			ret[i]=self.function_syn(s[i],self.sim_matrix[i])
		return ret

	def GetFunctionNormMatrix(self,s):
		ret=np.zeros((self.number,1))
		for i in range(0,self.number):
			ret[i]=self.function(s[i],1)
		ret=np.tile(ret,(1,len(s)))
		return ret

	def GetSumMatrix(self):
		ret=self.sim_matrix.sum(axis=1)
		ret=np.tile(ret,(self.sim_matrix.shape[0],1)).transpose()
		return ret

	def PrintStatus(self):
		openFile=open('case.txt','w',encoding='utf-8')

		key_word=u'急性后壁心肌梗死'

		word=u'急性心肌梗死'
		s=0.0
		for w1 in self.nodes:
			for w2 in self.nodes:
				tran1=self.weight_matrix[self.nodes_index[w1],self.nodes_index[key_word]]
				tran2=self.weight_matrix[self.nodes_index[w2],self.nodes_index[word]]
				sim=self.sim_matrix[self.nodes_index[w1],self.nodes_index[w2]]
				if w1!=w2 and tran1>10 and tran2>10 and sim>0:
					openFile.write(key_word+' '+str(tran1)+' '+w1+' '+str(sim)+' '+w2+' '+str(tran2)+' '+word+'\n')
					s+=self.trans_matrix[self.nodes_index[w1],self.nodes_index[key_word]]*sim*self.trans_matrix[self.nodes_index[w2],self.nodes_index[word]]
		openFile.write(str(self.sim_matrix[self.nodes_index[key_word],self.nodes_index[word]])+'+'+str(s)+'\n')
		openFile.write('\n')

		word=u'急性再发心肌梗死'
		s=0.0
		for w1 in self.nodes:
			for w2 in self.nodes:
				tran1=self.weight_matrix[self.nodes_index[w1],self.nodes_index[key_word]]
				tran2=self.weight_matrix[self.nodes_index[w2],self.nodes_index[word]]
				sim=self.sim_matrix[self.nodes_index[w1],self.nodes_index[w2]]
				if w1!=w2 and tran1>10 and tran2>10 and sim>0:
					openFile.write(key_word+' '+str(tran1)+' '+w1+' '+str(sim)+' '+w2+' '+str(tran2)+' '+word+'\n')
					s+=self.trans_matrix[self.nodes_index[w1],self.nodes_index[key_word]]*sim*self.trans_matrix[self.nodes_index[w2],self.nodes_index[word]]
		openFile.write(str(self.sim_matrix[self.nodes_index[key_word],self.nodes_index[word]])+'+'+str(s)+'\n')
		openFile.write('\n')



	def DrawNetwork(self, image_file):
		# 画图，暂时不太成功。。
		std_dic={}
		node_list,color_list,shell_list=[],[],[]
		openFile=open('xinjigengsi.txt',encoding='utf-8')
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split(' ')
			if items[1] not in std_dic:
				std_dic[items[1]]=[items[1]]
			if items[0] not in std_dic[items[1]]:
				std_dic[items[1]].append(items[0])
		openFile.close()

		g=nx.Graph()
		for key,value in std_dic.items():
			for name in value:
				if name not in node_list:
					node_list.append(name)
					color_list.append('red')
					g.add_node(name)
		shell_list.append(node_list)
		for name in node_list:
			for j in range(0,self.number):
				if self.weight_matrix[self.nodes_index[name],j]>20:
					g.add_edge(name,self.nodes[j])
		shell_list.append(list(set(g.nodes()).difference(set(node_list))))
		print(len(node_list),g.number_of_nodes(),g.number_of_edges())
		color_list.extend(['yellow']*(g.number_of_nodes()-len(node_list)))
		nx.draw_kamada_kawai(g,with_labels=True,node_size=100,font_size=4,node_color=color_list)
		plt.savefig(image_file,dpi=1000)
		#nx.draw_shell(g,nlist=shell_list,with_labels=True,node_size=200,font_size=4,node_color=color_list)
		#plt.savefig('2'+image_file,dpi=1000)

		

def FindDifference(case1, case2):
	file_name1='test_ans/ans'+str(case1)+'.txt'
	file_name2='test_ans/ans'+str(case2)+'.txt'
	file1=open(file_name1,encoding='utf-8')
	file2=open(file_name2,encoding='utf-8')
	openFile=open('diff.txt','w',encoding='utf-8')

	infered1={}
	lines=file1.readlines()
	for line in lines:
		items=re.split(':|\t',line.strip())
		infered1[items[0]]=(items[1],items[2])

	infered2={}
	lines=file2.readlines()
	for line in lines:
		items=re.split(':|\t',line.strip())
		infered2[items[0]]=(items[1],items[2])

	for key in infered1:
		if key not in infered2:
			continue
		# 0:a和b都正确；1:只有a正确；2:只有b正确；3:a和b都不正确，但是a=b；4:a和b都不正确，且a!=b
		if infered1[key][0]==infered1[key][1] and infered2[key][0]==infered1[key][1]:
			openFile.write(key+':'+infered1[key][0]+'\t'+infered2[key][0]+'\t'+infered1[key][1]+'\t0'+'\n')
		elif infered1[key][0]==infered1[key][1]:
			openFile.write(key+':'+infered1[key][0]+'\t'+infered2[key][0]+'\t'+infered1[key][1]+'\t1'+'\n')
		elif infered2[key][0]==infered1[key][1]:
			openFile.write(key+':'+infered1[key][0]+'\t'+infered2[key][0]+'\t'+infered1[key][1]+'\t2'+'\n')
		elif infered1[key][0]==infered2[key][0]:
			openFile.write(key+':'+infered1[key][0]+'\t'+infered2[key][0]+'\t'+infered1[key][1]+'\t3'+'\n')
		else:
			openFile.write(key+':'+infered1[key][0]+'\t'+infered2[key][0]+'\t'+infered1[key][1]+'\t4'+'\n')

	file1.close()
	file2.close()
	openFile.close()


if __name__=='__main__':
	sr=SimRank(0.1,10,'records.txt','i2025.txt')
	#sr.DrawNetwork('network.png')
	#sr.PrintStatus()
	sr.run_sim_rank()
	#sr.test_sim_rank('test.txt','test_ans/ans10.txt')
	#FindDifference(0,5)


	



