import matplotlib.pyplot as plt
import torch

u = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gat_differential_flow.pt', map_location='cpu')
v = torch.load('data_of_dataset1/output/pretrained_inference_model_for_dataset1_gat_flow.pt', map_location='cpu')

fig=plt.figure(figsize=(30.9,5), dpi=80)

x =range(192)

width = 0.8

y=u-v
y=torch.abs(y)
y=torch.mean(y,dim=1)

ax1 = fig.add_subplot(111)
ax1.bar(x,y,width=width, color="g",label="Prediction error")
ax1.legend(loc=2)

y1=u
y1=torch.mean(y1,dim=1)
y2=v
y2=torch.mean(y2,dim=1)

ax2 = ax1.twinx()
ax2.plot(x, y1, 'ro-', alpha=0.7, linewidth=2,markersize=2,label='average ${\hat F^n}$')
ax2.plot(x, y2, 'ys-', alpha=0.7, linewidth=2,markersize=2,label='average ${\hat F^d}$')
ax2.legend(loc=1)

ax1.set_xlabel('Time')  
ax1.set_ylabel('Prediction error')  
ax2.set_ylabel('Average flow')

x = range(192)
_x_label = ['0:00','6:00','12:00','18:00','0:00','6:00','12:00','18:00']
plt.xticks(list(x[::24]),_x_label)

plt.xlim(-1,193)
plt.show()








