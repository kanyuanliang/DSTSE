from torch.nn import Parameter
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from inference_model.inference_model_for_dataset2 import *
from fitting_model.fitting_model_for_dataset2 import *
from numerical_solution import *
from EarlyStopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Loaddata(Dataset):
	
	def __init__(self,data1):
		super().__init__()
		self.data1=data1
		self.len=len(data1)
	
	def __getitem__(self,index):
		return self.data1[index]
	
	def __len__(self):
		return self.len


train_data=torch.load('data_of_dataset2/input/flow_data.pt').to(device)
test_data=train_data[-192:]
train_data=train_data[:-192]

p_shape=torch.load('data_of_dataset2/input/p_shape.pt').to(device)
b_shape=torch.load('data_of_dataset2/input/b_shape.pt').to(device)
d_shape=torch.load('data_of_dataset2/input/d_shape.pt').to(device)
speed=torch.load('data_of_dataset2/input/speed.pt').to(device)

b_shape2=torch.ones(22,1).to(device)
b_shape2[0]=0
b_shape2[6]=0
b_shape2[12]=0
b_shape2[18]=0

u_shape2=torch.ones(22).to(device)
u_shape2[5]=0
u_shape2[11]=0
u_shape2[17]=0
u_shape2[21]=0

b_shape[0]=3.2
b_shape[6]=2
b_shape[12]=4.8
b_shape[18]=3.6

batch_size=512
train_data = Loaddata(train_data)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True)

critic_net=Critic().to(device)
actor_net=Actor(p_shape,b_shape,d_shape).to(device)

actor_optimizer=torch.optim.Adam(actor_net.parameters(),lr=1e-3)
critic_optimizer=torch.optim.Adam(critic_net.parameters(),lr=1e-3)
criterion=nn.MSELoss().to(device)

actor_path='data_of_dataset2/output/inference_model_for_dataset2_gat.pt'

state_dict = torch.load('data_of_dataset2/output/pretrained_fitting_model_for_dataset2.pt', map_location='cpu')
critic_net.load_state_dict(state_dict)
""
#actor_state_dict = torch.load(actor_path)
#actor_net.load_state_dict(actor_state_dict)

earlystopping = EarlyStopping(actor_path)

equation=Equation(batch_size,22,4,2,40)
test_equation=Equation(192,22,4,2,40)

t=0
min_test_loss=1e6
zero=torch.tensor(0).float().to(device)
one=torch.tensor(1).float().to(device)

while True:
	
	t+=1
	
	train_loss=[]
	
	for data in train_loader:
		
		actor_net.train()
		actor_optimizer.zero_grad()
		
		v,p,b,d,u=actor_net(data[:,:,:6])
		v2,p2,b2,d2,u2=actor_net(data[:,:,1:7])
		
		v_loss=criterion(v,one)
		v=v*speed
		
		net_flow=critic_net.net_untrained(torch.cat([v*p,u,b,d],2))
		
		with torch.no_grad():
			actor_net.eval()
			differential_u2=equation.last_u(v*p,torch.squeeze(u),torch.squeeze(b),torch.squeeze(d))
		
		actor_loss=criterion(data[:,:,6],net_flow)
		bd_loss=criterion(b*b_shape2,zero)+criterion(d,zero)
		u_loss=criterion(torch.squeeze(u2)*u_shape2,differential_u2*u_shape2)
		
		p_loss=torch.abs(p2-p)
		p_loss=torch.relu(p_loss-0.01)
		p_loss=torch.sum(p_loss)
		
		train_epoch_loss=actor_loss+bd_loss+1e6*torch.relu(v_loss-0.005)+5*u_loss+1e6*p_loss
		
		train_epoch_loss.backward()
		actor_optimizer.step()
		
		train_loss.append(train_epoch_loss)
	
	train_loss=torch.tensor(train_loss)
	train_loss=torch.mean(train_loss)
	
	if t%50==0:
		
		print("train_data results")
		print('train_loss','%.8f' % train_loss.item())
		print('actor_loss','%.8f' % actor_loss.item())
		print('bd_loss','%.8f' % bd_loss.item())
		print('u_loss','%.8f' % u_loss.item())
		print('v_loss','%.8f' % v_loss.item())
		print('p_loss','%.8f' % p_loss.item())
		print("")
	
	with torch.no_grad():
		actor_net.eval()
		
		v,p,b,d,u=actor_net(test_data[:,:,:6])
		v2,p2,b2,d2,u2=actor_net(test_data[:,:,1:7])
		
		v_loss=criterion(v,one)
		v=v*speed
		
		net_flow=critic_net.net_untrained(torch.cat([v*p,u,b,d],2))
		
		differential_u2=test_equation.last_u(v*p,torch.squeeze(u),torch.squeeze(b),torch.squeeze(d))
		
		actor_loss=criterion(test_data[:,:,6],net_flow)
		bd_loss=criterion(b*b_shape2,zero)+criterion(d,zero)
		u_loss=criterion(torch.squeeze(u2)*u_shape2,differential_u2*u_shape2)
		
		p_loss=torch.abs(p2-p)
		p_loss=torch.relu(p_loss-0.01)
		p_loss=torch.sum(p_loss)
		
		test_loss=actor_loss+bd_loss+1e6*torch.relu(v_loss-0.005)+5*u_loss+1e6*p_loss
		
		if test_loss<min_test_loss:
			
			min_test_loss=test_loss
			
			differential_flow,differential_u2=test_equation.last_output(v*p,torch.squeeze(u),torch.squeeze(b),torch.squeeze(d))
			differential_loss=criterion(differential_flow,net_flow)
			
			print("test_data results- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
			print('test_loss','%.8f' % test_loss.item())
			print('actor_loss','%.8f' % actor_loss.item())
			print('bd_loss','%.8f' % bd_loss.item())
			print('u_loss','%.8f' % u_loss.item())
			print('v_loss','%.8f' % v_loss.item())
			print('differential_loss','%.8f' % differential_loss.item())
			print('p_loss','%.8f' % p_loss.item())
			print("")
			torch.save(p,'data_of_dataset2/output/inference_model_for_dataset2_mlp_p.pt')
			torch.save(v,'data_of_dataset2/output/inference_model_for_dataset2_gat_v.pt')
			torch.save(b,'data_of_dataset2/output/inference_model_for_dataset2_gat_b.pt')
			torch.save(d,'data_of_dataset2/output/inference_model_for_dataset2_gat_d.pt')
			torch.save(u,'data_of_dataset2/output/inference_model_for_dataset2_gat_u.pt')
			torch.save(differential_u2,'data_of_dataset2/output/inference_model_for_dataset2_gat_differential_u2.pt')
			torch.save(u2,'data_of_dataset2/output/inference_model_for_dataset2_gat_u2.pt')
			torch.save(net_flow,'data_of_dataset2/output/inference_model_for_dataset2_gat_flow.pt')
			torch.save(differential_flow,'data_of_dataset2/output/inference_model_for_dataset2_gat_differential_flow.pt')

	earlystopping(test_loss,actor_net)
	if earlystopping.early_stop:
		print("Early stopping")
		break
	
	if t%4000==0:
		break
