import matplotlib.pyplot as plt
import torch


u1 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gat_u.pt', map_location='cpu')
u2 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gatv2_u.pt', map_location='cpu')
u3 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_dpgat_u.pt', map_location='cpu')

v1 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gat_v.pt', map_location='cpu')
v2 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gatv2_v.pt', map_location='cpu')
v3 = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_dpgat_v.pt', map_location='cpu')

t=9
u1,u2,u3=u1[:,t],u2[:,t],u3[:,t]
v1,v2,v3=v1[:,t],v2[:,t],v3[:,t]

fig=plt.figure(figsize=(30.9,5), dpi=80)

x_data = range(0,192,1)

ax1 = fig.add_subplot(111)
ax1.plot(x_data, u1, 'ro-', alpha=0.7, linewidth=1,markersize=1,label='$U_{i}(t)(t)$ by GAT')
ax1.plot(x_data, u2, 'yo-', alpha=0.7, linewidth=1,markersize=1,label='$U_{i}(t)$ by GATv2')
ax1.plot(x_data, u3, 'go-', alpha=0.7, linewidth=1,markersize=1,label='$U_{i}(t)$ by Multi-Head Attention')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(x_data, v1, 'rs--', alpha=0.7, linewidth=1,markersize=1,label='$V_{ii}$ by GAT')
ax2.plot(x_data, v2, 'ys--', alpha=0.7, linewidth=1,markersize=1,label='$V_{ii}$ by GATv2')
ax2.plot(x_data, v3, 'gs--', alpha=0.7, linewidth=1,markersize=1,label='$V_{ii}$ by Multi-Head Attention')
ax2.legend(loc=1)

ax1.set_xlabel('Time')  
ax1.set_ylabel('Number of vehicles in motion')  
ax2.set_ylabel('Vehicle speed')

x = range(192)
_x_label = ['0:00','6:00','12:00','18:00','0:00','6:00','12:00','18:00']
plt.xticks(list(x[::24]),_x_label)

plt.xlim(0,191)

plt.show()





