import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_b2
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from tqdm.auto import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
transforms_1 = weights.transforms()
effnetb2_model = efficientnet_b2(weights = weights)
auto_transform = transforms.Compose([transforms.TrivialAugmentWide(),
                                     transforms_1])
for params in effnetb2_model.parameters():
  params.requires_grad = False

effnetb2_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3,inplace = True),
    nn.Linear(in_features = 1408,
              out_features = 11)
)

train_data = ImageFolder(root = "Data/training",transform = auto_transform)
test_data = ImageFolder(root = "Data/validation",transform = auto_transform)

NUM_WORKERS = os.cpu_count()
NUM_WORKERS
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
optimizer = torch.optim.Adam(effnetb2_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

Epochs = 10

for epoch in tqdm(range(Epochs)):
  effnetb2_model.train()
  train_acc = 0
  train_loss = 0
  for x,y in train_dataloader:
    x,y = x.to(device), y.to(device)
    y_pred = effnetb2_model(x)
    pred_probs = torch.softmax(y_pred, dim= 1)
    pred_labels = torch.argmax(pred_probs, dim = 1)
    loss = loss_fn(y_pred,y)
    train_loss = train_loss + loss.item()
    train_acc += torch.sum(pred_labels == y).item()/len(y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss = train_loss/len(train_dataloader)
  train_acc = train_acc/len(train_dataloader)
  print('.........................................................................................................')
  test_loss,test_acc = 0,0
  for w,z in test_dataloader:
    w,z = w.to(device),z.to(device)
    effnetb2_model.eval()
    with torch.inference_mode():
      test_pred =effnetb2_model(w)
      test_pred_probs = torch.softmax(test_pred, dim = 1)
      test_label = torch.argmax(test_pred_probs , dim =1 )
      test_loss += loss_fn(test_pred,z)
      test_acc += torch.sum(test_label == z).item()/len(test_pred)
  test_acc = test_acc/len(test_dataloader)
  test_loss = test_loss/len(test_dataloader)
  print(f" Tain accuracy : {train_acc : 0.4f} | train loss : {train_loss : 0.2f} | test accuracy : {test_acc : 0.4f} | test loss: {test_loss : 0.2f}")