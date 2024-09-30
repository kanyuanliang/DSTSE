from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension):
		
		super(GATlayer,self).__init__()
		
		self.w=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.a=Parameter(torch.randn(output_dimension).to(device))

		self.leakyrelu = nn.LeakyReLU(0.01)

	def forward(self,x,A_shape):
		
		y=x@self.w
		x1=torch.unsqueeze(y,-2)
		x2=torch.unsqueeze(y,-3)
		
		A=x1+x2
		A=A@self.a
		
		A=torch.tanh(A/8)*8
		
		A=self.leakyrelu(A)
		
		A=torch.exp(A)*A_shape
		A=A/torch.unsqueeze(torch.sum(A,-1),-1)
		
		x=A@x
		x=x@self.w
		
		return x


class GNNlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension,num_heads=4):
		
		super(GNNlayer,self).__init__()
		
		self.attention_module = nn.ModuleList([GATlayer(input_dimension,output_dimension) for _ in range(num_heads)])
		
		self.linear= nn.Linear(output_dimension*num_heads,output_dimension)
	
	def forward(self,x,A_shape):
		
		x = torch.cat([attn(x,A_shape) for attn in self.attention_module], dim=-1)
		x = self.linear(x)
		
		return x
