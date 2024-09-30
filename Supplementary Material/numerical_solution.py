import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Equation:
		
	def __init__(self,B,d,n,l,g):
		
		self.n=n
		self.d=d
		self.l=l
		self.B=B
		self.g=g
		
		self.Rf_shape=torch.ones((self.d,self.d,self.d)).to(device)
		self.eye=torch.eye(self.d).to(device)
		
		self.A1_shape=torch.zeros((B,d,(n+1)*d,(n+1)*d)).to(device)
		
		self.A21_shape=torch.ones((B,d,(n+1)*d,d)).to(device)
		self.A22_shape=torch.zeros((B,d,d,(n+1)*d)).to(device)
		
		self.o1=torch.ones((self.d,1)).to(device)
		self.o2=torch.ones((self.n+1)).to(device)
		self.x=torch.tensor([1/(i+1) for i in range(self.n+1)]).to(device)
		""
		self.Rf_shape,self.Bf_shape=self.Rf_Bf_shape()
	
	def Rf_Bf_shape(self):
		Rf_shape=self.Rf_shape
		eye=self.eye
		for i in range(self.d):
			Rf_shape[i,:,i]=eye[:,i]
		Bf_shape=1-Rf_shape
		return Rf_shape,Bf_shape
	
	def R(self,P):#P:B*d*d
		V=torch.sum(P,dim=2)
		V=torch.unsqueeze(V,dim=2)
		V=V*self.eye*(-1)
		V=P+V+self.eye*(1e-5)*(-1)
		V=torch.unsqueeze(V,dim=1)
		return V
	
	def inverse_Rf_list(self,inverse_Rf,Bf):
		output=[]
		inverse_Rf,Bf=torch.unsqueeze(inverse_Rf,dim=1),torch.unsqueeze(Bf,dim=1)
		for i in range(self.n+1):
			if i == 0:
				inverse_Rf_k=inverse_Rf@Bf
			else:
				inverse_Rf_k=inverse_Rf@inverse_Rf_k
			output.append(inverse_Rf_k)
		output=torch.cat(output,dim=1) 
		return output
	
	def A1(self,inverse_Rf_list):
		output=self.A1_shape
		for i in range(self.n+1):
			for j in range(self.n+1):
				if j>=i:
					
					factorial=math.factorial(j)/math.factorial(i)
					
					output[:,:,i*self.d:(i+1)*self.d,j*self.d:(j+1)*(self.d)]=inverse_Rf_list[:,j-i].clone()*factorial
		return -1*output
	
	def A2(self,Rf,inverse_Rf_list):
		output1=self.A21_shape
		output2=self.A22_shape
		for i in range(self.n+1):
			if i == 0:
				output1[:,:,i*self.d:(i+1)*self.d,:]=(output1[:,:,i*self.d:(i+1)*self.d,:].clone())*self.eye
			elif i == 1:
				Rf_k=Rf
				output1[:,:,i*self.d:(i+1)*self.d,:]=Rf_k/math.factorial(i)
			else:
				Rf_k=Rf_k@Rf
				output1[:,:,i*self.d:(i+1)*self.d,:]=Rf_k/math.factorial(i)
			output2[:,:,:,i*self.d:(i+1)*self.d]=inverse_Rf_list[:,i].clone()*math.factorial(i)
		return output1,output1@output2
	
	def flow_metrix(self,R):
		
		Rf,Bf=self.Rf_shape*R,self.Bf_shape*R
		inverse_Rf=torch.inverse(Rf)
		
		inverse_Rf_list=self.inverse_Rf_list(inverse_Rf,Bf)
		A1=self.A1(inverse_Rf_list)
		last,A2=self.A2(Rf,inverse_Rf_list)
		
		A=A1+A2
		last=last@self.o1
		
		for i in range(self.l):
			if i == 0:
				l_k=A@last
				output=l_k*(i+1)
			else:
				l_k=A@l_k*(i+1)
				output+=l_k
		output=output.reshape(self.B,self.d,self.n+1,self.d)
		
		return output
	
	def flow_metrix2(self,p):
		p=p/self.g
		R=self.R(p)
		flow_metrix=self.flow_metrix(R)
		
		o=torch.unsqueeze(self.o2,0)
		o=torch.unsqueeze(o,0)
		
		x=torch.unsqueeze(self.x.float(),0)
		x=torch.unsqueeze(x,0)
		
		flow_metrix1=torch.squeeze(o@flow_metrix)
		flow_metrix2=torch.squeeze(x@flow_metrix)
		
		flow_metrix1=torch.unsqueeze(flow_metrix1,1)
		flow_metrix2=torch.unsqueeze(flow_metrix2,1)
		
		return torch.cat((flow_metrix1,flow_metrix2),1)

	def flow(self,output,b,d,u):
		
		u,b,d=torch.unsqueeze(u,dim=1),torch.unsqueeze(b,dim=1),torch.unsqueeze(d,dim=1)
		u,b,d=torch.unsqueeze(u,dim=3),torch.unsqueeze(b,dim=3),torch.unsqueeze(d,dim=3)
		
		flow1=output@u
		flow1=torch.squeeze(flow1)
		flow1=flow1@self.o2
		
		flow2=output@(b-d)
		flow2=torch.squeeze(flow2)
		flow2=flow2@(self.x.float())+torch.squeeze(b)
		return flow1+flow2

	def num_metrix(self,R):
		R=torch.squeeze(R)
		for i in range(self.n+1):
			if i == 0:
				R_k=self.eye
				output1=R_k/math.factorial(i)
				output2=R_k/math.factorial(i+1)
			else:
				R_k=R_k@R
				output1=output1+R_k/math.factorial(i)
				output2=output2+R_k/math.factorial(i+1)
		return output1,output2
	
	def num(self,output1,output2,u,b,d):
		
		u,b,d=torch.unsqueeze(u,dim=1),torch.unsqueeze(b,dim=1),torch.unsqueeze(d,dim=1)
		
		output1=u@output1
		output2=(b-d)@output2
		output1=torch.squeeze(output1)
		output2=torch.squeeze(output2)
		
		return output1+output2
	
	def last_u(self,p,u,b,d):
		
		p=p/self.g
		b=b/self.g
		d=d/self.g
		R=self.R(p)
		num_metrix1,num_metrix2=self.num_metrix(R)
		for i in range(self.g*1):
			u=self.num(num_metrix1,num_metrix2,u,b,d)
		return u

	def last_output(self,p,u,b,d):
		
		p=p/self.g
		b=b/self.g
		d=d/self.g
		R=self.R(p)
		num_metrix1,num_metrix2=self.num_metrix(R)
		flow_metrix=self.flow_metrix(R)
		for i in range(self.g*1):
			flow=self.flow(flow_metrix,b,d,u)
			u=self.num(num_metrix1,num_metrix2,u,b,d)
			if i == 0:
				total_flow=flow
			else:
				total_flow=total_flow+flow
		return total_flow,u
