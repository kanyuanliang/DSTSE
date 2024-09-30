import math
import torch
from numerical_solution import * # import package of numerical solution

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def truncated_normal_distribution(B,d,interval):
	
	data1=torch.randn(B,d,1).to(device)
	data2=torch.randn(B,d,1).to(device)
	data3=(torch.rand(B,d,1).to(device))*(interval[1]-interval[0])+interval[0]
	
	data1_shape=torch.where(data1<interval[1],1,0)+torch.where(data1>interval[0],1,0)
	data1_shape=torch.where(data1_shape==2,1,0)
	
	data=data1*(data1_shape)+data2*(1-data1_shape)
	
	data_shape=torch.where(data<interval[1],1,0)+torch.where(data>interval[0],1,0)
	data_shape=torch.where(data_shape==2,1,0)
	
	data=data*(data_shape)+data3*(1-data_shape)
	return data


def select_data(data,max_data):
	
	flow=data[:,:,-1]
	
	index1=max_data-flow+100
	index1=torch.min(index1,dim=1)[0]
	index1=torch.where(index1>=0,1,0)
	
	index2=torch.min(flow,dim=1)[0]
	index2=torch.where(index2>=0,1,0)
	
	index=index1+index2
	
	data=data[index==2]
	return data


def collect_train_data(B,d,speed,p_shape,b_shape,d_shape,max_data):
	
	pra=Equation(B,d,4,2,40)
	output=[]
	
	for i in range(400):
		
		if i%40==0:
			print(i)
		
		speed_distribution=truncated_normal_distribution(B,d,[-3,0.5])
		speed_distribution=speed_distribution/6+1
		speed_distribution=speed_distribution*speed
		
		p_distribution=torch.rand(B,d,d).to(device)
		p_distribution=p_distribution*p_shape
		p_distribution=p_distribution/(torch.unsqueeze(torch.sum(p_distribution,dim=-1),-1)+1e-15)
		
		batch_vp=speed_distribution*p_distribution
		
		batch_b1=torch.rand(B,1).to(device)*1600
		batch_b2=(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*300
		batch_b3=torch.rand(B,1).to(device)*1000
		batch_b4=(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*300
		batch_b5=torch.rand(B,1).to(device)*2400
		batch_b6=(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*(torch.rand(B,5).to(device))*300
		batch_b7=torch.rand(B,1).to(device)*1800
		batch_b8=(torch.rand(B,3).to(device))*(torch.rand(B,3).to(device))*(torch.rand(B,3).to(device))*(torch.rand(B,3).to(device))*300
		batch_b=[batch_b1,batch_b2,batch_b3,batch_b4,batch_b5,batch_b6,batch_b7,batch_b8]
		batch_b=torch.cat(batch_b,1)*b_shape
		
		batch_d=(torch.rand(B,d).to(device))*(torch.rand(B,d).to(device))*(torch.rand(B,d).to(device))*d_shape*300
		batch_d[:,5]=0
		batch_d[:,11]=0
		batch_d[:,17]=0
		batch_d[:,21]=0
		
		batch_u=(torch.rand(B,d).to(device))*(torch.rand(B,d).to(device))*(torch.rand(B,d).to(device))*500
		batch_u[:,5]=0
		batch_u[:,11]=0
		batch_u[:,17]=0
		batch_u[:,21]=0
		
		flow,num=pra.last_output(batch_vp,batch_u,batch_b,batch_d)
		
		batch_u,batch_b,batch_d=torch.unsqueeze(batch_u,2),torch.unsqueeze(batch_b,2),torch.unsqueeze(batch_d,2)
		flow=torch.unsqueeze(flow,2)
		segment=torch.cat([batch_vp,batch_u,batch_b,batch_d,flow],2)
		
		segment=select_data(segment,max_data)
		
		output.append(segment)
	
	output=torch.cat(output,0)
	print('Sample_size: ',len(output))
	
	return output


B=320 #Number of samples extracted per iteration
d=22 #Number of intersections

p_shape=torch.load('data_of_dataset2/input/p_shape.pt').to(device)
b_shape=torch.load('data_of_dataset2/input/b_shape.pt').to(device)
d_shape=torch.load('data_of_dataset2/input/d_shape.pt').to(device)
max_data=torch.load('data_of_dataset2/input/max_data.pt').to(device)
speed=torch.load('data_of_dataset2/input/speed.pt').to(device)
b_shape=torch.squeeze(b_shape)
d_shape=torch.squeeze(d_shape)

output=collect_train_data(B,d,speed,p_shape,b_shape,d_shape,max_data)
torch.save(output,'data_of_dataset2/output/traindata_for_fitting_model2.pt')


