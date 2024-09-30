import matplotlib.pyplot as plt
import torch

p1 = torch.load('data_of_dataset2/output/pretrained_inference_model_for_dataset2_gat_p.pt', map_location='cpu')
p2 = torch.load('data_of_dataset2/output/pretrained_inference_model_for_dataset2_gatv2_p.pt', map_location='cpu')
p3 = torch.load('data_of_dataset2/output/pretrained_inference_model_for_dataset2_dpgat_p.pt', map_location='cpu')

a1=20
a2=9
p1,p2,p3=p1[:,a1,a2],p2[:,a1,a2],p3[:,a1,a2]

x=192
x=range(x)

fig=plt.figure(figsize=(30.9,5), dpi=80)

plt.plot(x, p1, 'rs--', alpha=0.7, linewidth=1,markersize=1,label='$P_{ij}$ by GAT')
plt.plot(x, p2, 'ys--', alpha=0.7, linewidth=1,markersize=1,label='$P_{ij}$ by GATv2')
plt.plot(x, p3, 'gs--', alpha=0.7, linewidth=1,markersize=1,label='$P_{ij}$ by Multi-Head Attention')

x = range(192)
_x_label = ['0:00','6:00','12:00','18:00','0:00','6:00','12:00','18:00']
plt.xticks(list(x[::24]),_x_label)

plt.xlim(0,191)

plt.xlabel('Time')
plt.ylabel('Probability')

plt.legend()

plt.show()



















