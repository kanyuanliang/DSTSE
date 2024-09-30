from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DPGATlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension):
		
		super(DPGATlayer,self).__init__()
		
		self.q=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.k=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.v=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		
		self.leakyrelu = nn.LeakyReLU(0.01)
		
		self.input_dimension=input_dimension

	def forward(self,x,A_shape):
		
		Q=x@self.q
		K=x@self.k
		A=Q@(K.transpose(-2,-1))
		A=A/(self.input_dimension**(0.5))
		
		A=torch.tanh(A/8)*8
		
		A=torch.exp(A)*A_shape
		A=A/torch.unsqueeze(torch.sum(A,-1),-1)
		
		x=A@x
		x=x@self.v
		
		return x


class GNNlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension,num_heads=4):
		
		super(GNNlayer,self).__init__()
		
		self.attention_module = nn.ModuleList([DPGATlayer(input_dimension,output_dimension) for _ in range(num_heads)])
		
		self.linear= nn.Linear(output_dimension*num_heads,output_dimension)
		
	def forward(self,x,A_shape):
		
		x = torch.cat([attn(x,A_shape) for attn in self.attention_module], dim=-1)
		x = self.linear(x)
		
		return x
