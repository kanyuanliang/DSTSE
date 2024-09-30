import torch
from torch.nn import Parameter
import torch.nn as nn


class Critic(nn.Module):
	
	def __init__(self,linear1=[23,20,15,15,15,10,5,1],linear2=[int(23*4),int(23*4),int(23*3),int(23*3),int(23*3),int(23*3),int(23*2),int(23*2),int(23*1)]):
		
		super(Critic, self).__init__()
		self.unit_1=nn.ModuleList([nn.Linear(linear1[i],linear1[i+1]) for i in range(len(linear1)-1)])
		self.unit_2=nn.ModuleList([nn.Linear(linear1[i],linear1[i+1]) for i in range(len(linear1)-1)])
		self.unit_3=nn.ModuleList([nn.Linear(linear2[i],linear2[i+1]) for i in range(len(linear2)-1)])
		self.unit_4=nn.ModuleList([nn.Linear(linear2[i],linear2[i+1]) for i in range(len(linear2)-1)])
		
		self.leakyrelu = nn.LeakyReLU(0.01)
	
	def net1(self,x,layers1,layers2):
		
		p=x[:,:,:23]
		
		for i,layer in enumerate(layers1):
			p=layer(p)
			p=self.leakyrelu(p)
		p=torch.squeeze(p)
		
		x=x[:,:,-3:]
		x=x.transpose(2,1)
		x=x.contiguous().view(-1,int(3*23))
		x=torch.cat((p,x),1)
		
		for i,layer in enumerate(layers2):
			x=layer(x)
			x=self.leakyrelu(x)
		
		return x
	
	def net2(self,x,layers1,layers2):
		
		p=x[:,:,:23]
		p=p.transpose(2,1)
		
		for i,layer in enumerate(layers1):
			p=layer(p)
			p=self.leakyrelu(p)
		p=torch.squeeze(p)
		
		x=x[:,:,-3:]
		x=x.transpose(2,1)
		x=x.contiguous().view(-1,int(3*23))
		x=torch.cat((p,x),1)
		
		for i,layer in enumerate(layers2):
			x=layer(x)
			x=self.leakyrelu(x)
		
		return x
	
	def net_untrained(self,x):
		
		x1=self.net1(x,self.unit_1,self.unit_3)
		x2=self.net2(x,self.unit_2,self.unit_4)
		
		return x1+x2
	
	def forward(self,x):
		
		x1=self.net1(x,self.unit_1,self.unit_3)
		x2=self.net2(x,self.unit_2,self.unit_4)
		
		return x1+x2
