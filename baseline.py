import numpy as np
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import pytorch_lightning as pl
#from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd

train = pd.read_table("./data/train.txt",header=None, sep="|")
valid = pd.read_table("./data/valid.txt",header=None, sep="|")
test = pd.read_csv("./data/eval.txt",header=None, sep="|")

label_list = ['X','T','O']
label_dict = dict()
for i in label_list:
  label_dict[i] = len(label_dict)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_len = 256
batch_size = 32

def make_dataset(tokenizer, max_length, data):
    dataset_for_loader = list()
    
    for idx,line in data.iterrows():
        # ラベルとテキストを読み込み

        # テキストをトークンに分割する。ただし、最大文長は "max_length" で指定したトークン数である。
        # 最大文長より短い文については、 "[PAD]" などの特殊トークンで残りの長さを埋める。
        # 最大文長を超える文については、はみ出す部分を無視する。
        encoding = tokenizer(line[0], line[1], max_length=max_length, padding="max_length", truncation=True)

        # tokenizerメソッドは辞書を返す。その辞書にラベルのIDも持たせる。
        encoding["labels"] = label_dict[line[2]]

        # テンソルに変換
        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        # 前処理済みのデータを保存して次の文へ
        dataset_for_loader.append(encoding)

    return dataset_for_loader

dataset_train = make_dataset(tokenizer, max_len, train)
dataset_valid = make_dataset(tokenizer, max_len, valid)
dataset_test = make_dataset(tokenizer, max_len, test)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=256, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

class BertForSequenceClassification_pl(pl.LightningModule):
  def __init__(self, model, num_labels, lr):
    super().__init__()
    self.save_hyperparameters()
    self.bert_sc = BertForSequenceClassification.from_pretrained(
      model, num_labels=num_labels
    )

  def training_step(self,batch,batch_idx):
    output = self.bert_sc(**batch)
    loss = output.loss
    self.log('train_loss',loss)
    return loss

  def validation_step(self,batch,batch_idx):
    output = self.bert_sc(**batch)
    val_loss = output.loss
    self.log('val_loss', val_loss)

  def test_step(self,batch,batch_idx):
    labels = batch.pop('labels')
    output = self.bert_sc(**batch)
    labels_predicted = output.logits.argmax(-1)
    num_correct = (labels_predicted == labels).sum().item()
    print(num_correct)
    accuracy = num_correct/labels.size(0)
    self.log('accuracy',accuracy)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr = self.hparams.lr)

checkpoint = pl.callbacks.ModelCheckpoint(
  monitor='val_loss',
  mode='min',
  save_top_k=1,
  save_weights_only=True,
  dirpath='./models'
)

model = BertForSequenceClassification_pl(
  model_name, num_labels=3, lr=2e-5
)

trainer = pl.Trainer(
  gpus=1,
  max_epochs=10,
  callbacks = [checkpoint]
)

early_stop_callback = EarlyStopping(
    patience=3,
    monitor='val_loss',
    mode='min',
)

trainer.fit(model, dataloader_train, dataloader_valid)

print('ベストモデルのファイル：', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失：', checkpoint.best_model_score)

test = trainer.test(test_dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.2f}')
