from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATv2layer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension):
		
		super(GATv2layer,self).__init__()
		
		self.w1=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.w2=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.a=Parameter(torch.randn(output_dimension).to(device))
		
		self.linear=nn.Linear(input_dimension,output_dimension,bias=False)
		
		self.leakyrelu = nn.LeakyReLU(0.01)

	
	def forward(self,x,A_shape):
		
		x1=x@self.w1
		x2=x@self.w2
		x1=torch.unsqueeze(x1,-2)
		x2=torch.unsqueeze(x2,-3)
		
		A=x1+x2
		
		A=self.leakyrelu(A)
		A=A@self.a
		
		A=torch.tanh(A/8)*8
		
		A=torch.exp(A)*A_shape
		A=A/torch.unsqueeze(torch.sum(A,-1),-1)
		
		x=A@x
		x=self.linear(x)
		x=self.leakyrelu(x)
		
		return x


class GNNlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension,num_heads=4):
		
		super(GNNlayer,self).__init__()
		
		self.attention_module = nn.ModuleList([GATv2layer(input_dimension,output_dimension) for _ in range(num_heads)])
		
		self.w= Parameter(torch.randn(output_dimension*num_heads,output_dimension).to(device))
		
	def forward(self,x,A_shape):
		
		x = torch.cat([attn(x,A_shape) for attn in self.attention_module], dim=-1)
		x = x@self.w
		
		return x
