import numpy as np
import csv
import matplotlib.pyplot as plt

path1='./training_log_cnn_pretrain.csv'
with open(path1,'r') as f:
    reader1 = csv.reader(f)
    datas1 = [[row[0],row[1],row[2],row[3],row[4]] for row in reader1]

datas1=np.array(datas1[1:31],dtype=np.float32)


path2='./training_log_cnn_custom.csv'
with open(path2,'r') as f:
    reader2 = csv.reader(f)
    datas2 = [[row[0],row[1],row[2],row[3],row[4]]  for row in reader2]

datas2=np.array(datas2[1:31],dtype=np.float32)

x = np.array(range(1,31))
plt.figure()
plt.rcParams["font.family"] = "Arial"
lw=1.5

ax1 = plt.subplot(2,1,1)
plt.plot(x, datas1[:,0], color='darkorange',
         linewidth=lw, label='Anomaly accuracy = %0.2f' % 0.88)
plt.plot(x, datas1[:,1], color='blue',
         linewidth=lw, label='Anomaly precision = %0.2f' % 0.85)
plt.plot(x, datas1[:,2], color='red',
         linewidth=lw, label='Anomaly recall = %0.2f' % 0.84)
plt.plot(x, datas1[:,3], color='black',
         linewidth=lw, label='Anomaly F1_score = %0.2f' % 0.84)
plt.plot(x, datas1[:,4], color='green',
         linewidth=lw, label='Pattern accuracy = %0.2f' % 0.96)

plt.xlim([1, 31])
plt.ylim([0.3, 1.0])

plt.xlabel('Number of epoch')
plt.ylabel('Evaluation metrics')
plt.legend(loc="lower right")
ax = plt.gca()
#ax.set_aspect(1)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

ax2 = plt.subplot(2,1,2)

plt.plot(x, datas2[:,0], color='darkorange',
         linewidth=lw, label='Anomaly accuracy = %0.2f' % 0.85)
plt.plot(x, datas2[:,1], color='blue',
         linewidth=lw, label='Anomaly precision = %0.2f' % 0.81)
plt.plot(x, datas2[:,2], color='red',
         linewidth=lw, label='Anomaly recall = %0.2f' % 0.79)
plt.plot(x, datas2[:,3], color='black',
         linewidth=lw, label='Anomaly F1_score = %0.2f' % 0.80)
plt.plot(x, datas2[:,4], color='green',
         linewidth=lw, label='Pattern accuracy = %0.2f' % 0.94)

plt.xlim([1, 31])
plt.ylim([0.3, 1.0])

plt.xlabel('Number of epoch')
plt.ylabel('Evaluation metrics')
plt.legend(loc="lower right")
ax = plt.gca()
#ax.set_aspect(1)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.show()