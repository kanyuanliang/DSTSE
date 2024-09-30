import torch
from torch.nn import Parameter
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import math
from fitting_model.fitting_model_for_dataset1 import *
from EarlyStopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Loaddata(Dataset):
	""
	def __init__(self,data1):
		super().__init__()
		self.data1=data1
		self.len=len(data1)
	""
	def __getitem__(self,index):
		return self.data1[index]
	""
	def __len__(self):
		return self.len


train_data=torch.load('data_of_dataset1/output/preextracted_traindata_for_fitting_model1.pt', map_location='cpu').to(device)
test_data=train_data[-4096:]
train_data=train_data[:-4096]

batch_size=2048
train_data = Loaddata(train_data)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

net=Critic().to(device)
optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
criterion=nn.MSELoss().to(device)

path='data_of_dataset1/output/fitting_model_for_dataset1.pt'

earlystopping = EarlyStopping(path)
# import the pre-trained model
#state_dict = torch.load(path)
#net.load_state_dict(state_dict)

i=0
min_test_loss=1e6

while True:
	
	net.train()
	
	i+=1
	train_loss=0
	
	for data in train_loader:
		
		optimizer.zero_grad()
		
		x=data[:,:,:-1]
		y=data[:,:,-1]
		
		y_prediction=net(x)
		
		train_epoch_loss=criterion(y_prediction,y)
		loss=train_epoch_loss
		
		loss.backward()
		optimizer.step()
		
		train_loss+=train_epoch_loss*len(data)
		
	train_loss=train_loss/len(train_data)
	
	if i%1==0:
		with torch.no_grad():
			net.eval()
			
			x=test_data[:,:,:-1]
			y=test_data[:,:,-1]
			y_prediction=net(x)
			test_loss=criterion(y_prediction,y)
			
			print("train_loss: ",'%.8f' % train_loss.item())
			print("test_loss: ",'%.8f' % test_loss.item())
			print("")
	
	if test_loss<min_test_loss:
		min_test_loss=test_loss
		torch.save(net.state_dict(),path)
	
	if i==1000:
		break
