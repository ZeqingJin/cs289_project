import numpy as np
import csv


path1='./testing_result_pretrain.csv'
with open(path1,'r') as f:
    reader1 = csv.reader(f)
    datas1 = [[row[5],row[6],row[7],row[8],row[9],row[10]] for row in reader1]

datas1=np.array(datas1[1:],dtype=np.float32)
labels1=datas1[:,[0,1,2]]
predicts1=datas1[:,[3,4,5]]

path2='./testing_result_custom.csv'
with open(path2,'r') as f:
    reader2 = csv.reader(f)
    datas2 = [[row[5],row[6],row[7],row[8],row[9],row[10]] for row in reader2]

datas2=np.array(datas2[1:],dtype=np.float32)
labels2=datas2[:,[0,1,2]]
predicts2=datas2[:,[3,4,5]]




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


n_classes=3
fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(labels1[:, i], predicts1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(labels2[:, i], predicts2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])


plt.figure()
plt.rcParams["font.family"] = "Arial"
ax1 = plt.subplot(2,1,1)
plt.plot(fpr1[0], tpr1[0], color='darkorange',
         linewidth=2, label='ROC curve of anomaly 1 (area = %0.2f)' % roc_auc1[0])
plt.plot(fpr1[1], tpr1[1], color='blue',
         linewidth=2, label='ROC curve of anomaly 2 (area = %0.2f)' % roc_auc1[1])
plt.plot(fpr1[2], tpr1[2], color='red',
         linewidth=2, label='ROC curve of anomaly 3 (area = %0.2f)' % roc_auc1[2])

#plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.4, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_aspect(0.8)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

ax2 = plt.subplot(2,1,2)
plt.plot(fpr2[0], tpr2[0], color='darkorange',
         linewidth=2, label='ROC curve of anomaly 1 (area = %0.2f)' % roc_auc2[0])
plt.plot(fpr2[1], tpr2[1], color='blue',
         linewidth=2, label='ROC curve of anomaly 2 (area = %0.2f)' % roc_auc2[1])
plt.plot(fpr2[2], tpr2[2], color='red',
         linewidth=2, label='ROC curve of anomaly 3 (area = %0.2f)' % roc_auc2[2])

#plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.4, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_aspect(0.8)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)


plt.show()