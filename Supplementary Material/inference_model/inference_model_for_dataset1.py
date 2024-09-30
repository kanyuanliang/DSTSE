from torch.nn import Parameter
import torch
import torch.nn as nn

from inference_model.gnn_model.GAT import *
#from inference_model.gnn_model.GATv2 import *
#from inference_model.gnn_model.DPGAT import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sub_Actor(nn.Module):
	
	def __init__(self,linear1,linear2,gnn1):
		
		super(Sub_Actor,self).__init__()
		
		n=23
		
		self.len_linear2=len(linear2)
		
		self.gnn_list=nn.ModuleList([GNNlayer(gnn1[i],gnn1[i+1]) for i in range(len(gnn1)-1)])
		self.linear1_list=nn.ModuleList([nn.Linear(linear1[i],linear1[i+1]) for i in range(len(linear1)-1)])
		self.linear2_list=nn.ModuleList([nn.Linear(linear2[i],linear2[i+1]) for i in range(len(linear2)-1)])
		
		len_para=len(linear1)+len(linear2)+len(gnn1)-3
		self.parameter=Parameter(torch.rand((len_para,n,1)))
		
		self.leakyrelu = nn.LeakyReLU(0.01)
	
	def forward(self,x,A_shape):
		
		t=0
		
		for i,layer in enumerate(self.linear1_list):
			x=layer(x)
			x=self.leakyrelu(x)
			x=x*(self.parameter[t])
			t+=1
		
		for i,layer in enumerate(self.gnn_list):
			x=layer(x,A_shape)
			x=self.leakyrelu(x)
			x=x*(self.parameter[t])
			t+=1
		
		for i,layer in enumerate(self.linear2_list):
			x=layer(x)
			if i<(self.len_linear2-2):
				x=self.leakyrelu(x)
			x=x*(self.parameter[t])
			t+=1
		
		return x


class Actor(nn.Module):
	
	def __init__(self,p_shape,b_shape,d_shape,\
	n=23,linear1=[6,4,4,4],linear2=[4,4,2,1],linear3=[4,4,4,4],gnn1=[4,4]):
		
		super(Actor,self).__init__()
		
		self.p_shape=p_shape
		self.b_shape=b_shape
		self.d_shape=d_shape
		last_shape=torch.ones(23,1)
		last_shape[-1]=0
		self.last_shape=last_shape.to(device)
		
		self.A_shape=p_shape+p_shape.transpose(1,0)+torch.eye(23).to(device)
		
		self.unit_p1=Sub_Actor(linear1,linear3,gnn1)
		self.unit_p2=Sub_Actor(linear1,linear3,gnn1)
		
		self.unit_b=Sub_Actor(linear1,linear2,gnn1)
		self.unit_d=Sub_Actor(linear1,linear2,gnn1)
		self.unit_u=Sub_Actor(linear1,linear2,gnn1)
		self.unit_v1=Sub_Actor(linear1,linear2,gnn1)
		self.unit_v2=Sub_Actor([3,4,4,4],[4,4,2,1],gnn1)
		
		self.leakyrelu = nn.LeakyReLU(0.01)
	
	def forward(self,x):
		
		p1=self.unit_p1(x,self.A_shape)
		p2=self.unit_p2(x,self.A_shape)
		
		u=self.unit_u(x,self.A_shape)
		b=self.unit_b(x,self.A_shape)
		d=self.unit_d(x,self.A_shape)
		
		b,d,u=torch.abs(b),torch.abs(d),torch.abs(u)
		b,d,u=torch.where(b<450,b,0.01*b+445.5),torch.where(d<200,d,0.01*d+198),torch.where(u<800,u,0.01*u+792)
		b,d=b*self.b_shape,d*self.d_shape
		
		d,u=d*self.last_shape,u*self.last_shape
		
		v=torch.cat([u,b,d],dim=-1)
		v=self.unit_v2(v,self.A_shape)
		v=torch.tanh(v/100)*0.4+1
		
		return v,self.p_shape,b,d,u
