import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch.utils.data import DataLoader
from models import botnet, Classfication
from intrusion_detection_datasets import NSL_KDDDataset, KDD_CUPDataset, CIC_IDS2017Dataset, ISCX_IDS2012Dataset, CIDDS_001Dataset


epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = botnet([32, 64, 128, 256]).to(device)
labled_model = BoTNet_with_classifier(BoTNetBottleneck, [2, 2, 2, 2], num_classes=1).to(device)
# 二分类
c_net = Classfication(512, 1).to(device)
model.load_state_dict(torch.load('Representation.pt'))
for param in model.parameters():
    param.requires_grad = False

train_csv_path = './datasets/KDDTrain+.txt'
test_csv_path = './datasets/KDDTest+.txt'
ds = NSL_KDDDataset(train_csv_path, test_csv_path, 14)


ds_length = len(ds)
train_size = int(0.8 * ds_length)
val_size = ds_length - train_size
# 切分验证数据集
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
test_size = int(0.15*train_size)
train_size = train_size - test_size
# 切分测试数据集
train_ds, test_ds = torch.utils.data.random_split(train_ds, [train_size, test_size])


train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=True, drop_last=True)
test_size -= test_size % 128


criterion = torch.nn.BCELoss()
criterion_with_classifier = torch.nn.BCELoss()

optimizer = torch.optim.Adam(c_net.parameters(), lr=1e-3)
optimizer_with_classifier = torch.optim.Adam(labled_model.parameters(), lr=1e-3)

total_step = len(train_loader)
for epoch in range(epochs):
    c_net.train()
    labled_model.train()
    for ids, (data, target) in enumerate(train_loader):
        torch.cuda.empty_cache()
        data = data.to(device)
        target = target.to(device)

        # forward
        logits = c_net(torch.squeeze(model(data)))
        logits = torch.squeeze(logits, dim=1)
        loss = criterion(logits, target.float())

        logits_with_classifier = labled_model(data)
        logits_with_classifier = torch.squeeze(logits_with_classifier, dim=1)
        loss_with_classifier = criterion_with_classifier(logits_with_classifier, target.float())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_with_classifier.zero_grad()
        loss_with_classifier.backward()
        torch.nn.utils.clip_grad_norm_(labled_model.parameters(), 1, norm_type=2)
        optimizer_with_classifier.step()

        if (ids + 1) % 200 == 0:
            print('Transfer learning Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch+1, epochs, ids + 1, total_step, loss.item()))
            print('Classification: Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch + 1, epochs, ids + 1,
                                                                                     total_step,
                                                                                     loss_with_classifier.item()))



测试数据集
num_correct = 0
num_samples = 0
num_correct_with_classifier = 0
y_test = torch.zeros(size=(test_size,)).to(device)
y_predict = torch.zeros(size=(test_size,)).to(device)
y_test_with_classifier = torch.zeros(size=(test_size, )).to(device)
y_predict_with_classifier = torch.zeros(size=(test_size, )).to(device)
c_net.eval()
labled_model.eval()

with torch.no_grad():
    for index, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        y_test[index * 128:(index + 1) * 128] = y
        y_test_with_classifier[index * 128:(index + 1) * 128] = y

        logits = c_net(torch.squeeze(model(x)))
        logits_with_classifier = labled_model(x)

        prediction = torch.where(logits > 0.5, 1, 0)
        prediction = torch.squeeze(prediction, dim=1)
        y_predict[index * 128:(index + 1) * 128] = prediction

        prediction_with_classifier = torch.where(logits_with_classifier > 0.5, 1, 0)
        prediction_with_classifier = torch.squeeze(prediction_with_classifier, dim=1)
        y_predict_with_classifier[index * 128:(index + 1) * 128] = prediction_with_classifier
        num_correct_with_classifier += (y == prediction_with_classifier).sum()

        num_correct += (y == prediction).sum()
        num_samples += y.size(0)

    print(f'Finally, Transfer learning got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples):.4f}')
    print(f'Finally, Classification: got {num_correct_with_classifier} / {num_samples} with accuracy {float(num_correct_with_classifier) / float(num_samples):.4f}')

y1 = y_test.cpu().numpy()
y2 = y_predict.cpu().numpy()

y1_with_classifier = y_test_with_classifier.cpu().numpy()
y2_with_classifier = y_predict_with_classifier.cpu().numpy()


fpr, tpr, threshold = roc_curve(y1, y2)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值

fpr_with_classifier, tpr_with_classifier, threshold_with_classifier = roc_curve(y1_with_classifier, y2_with_classifier)  ###计算真正率和假正率
roc_auc_with_classifier = auc(fpr_with_classifier, tpr_with_classifier)  ###计算auc的值


plt.figure(dpi=300)
plt.plot(fpr_with_classifier, tpr_with_classifier, color='blue', label='Supervised BoTNet ROC curve (area = %0.2f)' % roc_auc_with_classifier)
plt.plot(fpr, tpr, color='black', label='Self-Supervised Learning ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Supervised BoTNet and Self-Supervised Learning ROC curve')
plt.legend(loc="lower right")
plt.show()


print("Transfer learning accuracy is", accuracy_score(y1, y2))
print("Transfer learning precision is", precision_score(y1, y2))
print("Transfer learning recall is", recall_score(y1, y2))
print("Transfer learning f1 score is", f1_score(y1, y2))

print("Classification accuracy is", accuracy_score(y1_with_classifier, y2_with_classifier))
print("Classification precision is", precision_score(y1_with_classifier, y2_with_classifier))
print("Classification recall is", recall_score(y1_with_classifier, y2_with_classifier))
print("Classification f1 score is", f1_score(y1_with_classifier, y2_with_classifier))
