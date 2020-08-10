import argparse
import sys
import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import SentenceDataset
from metric import Metric
from loss import FocalLoss
from alarm_bot import ExamAlarmBot

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=130)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--warmup_epoch_count', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=34)
parser.add_argument('--bert_model_name', type=str, default='beomi/kcbert-large')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--data_dir', type=str, default='data')
args = parser.parse_args()


# load params
batch_size = args.batch_size
num_epochs = args.num_epochs
num_warmup_epochs = args.warmup_epoch_count
data_path = args.data_dir
result_path = args.result_dir
train_name = "ser_{}_{}_{}".format(int(args.alpha * 100), int(args.gamma * 10), datetime.now().strftime("%m%d-%H%M"))
train_path = os.path.join(result_path, train_name)
if not os.path.exists(train_path):
    os.mkdir(train_path)


# dataloader
train_dataset = SentenceDataset(os.path.join(data_path, 'trainsets.pkl'))
val_dataset = SentenceDataset(os.path.join(data_path, 'valsets.pkl'))
test_dataset = SentenceDataset(os.path.join(data_path, 'testsets.pkl'))

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

num_steps_per_epochs = {
    'train': len(train_dataset) // batch_size,
    'val': len(val_dataset) // batch_size,
    'test': len(test_dataset) // batch_size
}

print(dataset_sizes)
num_steps_per_epoch = dataset_sizes['train'] // batch_size
num_train_steps = num_steps_per_epoch * num_epochs
num_warmup_steps = num_steps_per_epoch * num_warmup_epochs

# init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=args.num_classes)
model.train()
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
writer = SummaryWriter(log_dir=os.path.join(train_path, 'logs'))

print('batch_size {}, num_epochs {}, step_per_epoch {}'.format(batch_size, num_epochs, num_steps_per_epoch))
eval_history = []

# train loop
for epoch in range(num_epochs):
    print('Train Epoch {} / {}'.format(epoch+1, num_epochs))

    for phase in ['train', 'val', 'test']:
        if phase == 'test' and epoch + 1 != num_epochs:
            continue

        if phase == 'train':
            model.train()  # 모델을 학습 모드로 설정
        else:
            model.eval()   # 모델을 평가 모드로 설정

        metric = Metric()
        start = time.time()
        running_loss = 0
        for step, dataset in enumerate(dataloaders[phase]):
            # data
            input_ids = dataset['inputs'].to(device)
            attention_mask = dataset['mask'].to(device)
            labels = dataset['labels'].type(torch.FloatTensor).to(device)

            # train model
            optimizer.zero_grad()
            model.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs[0])
            loss = criterion(preds, labels)

            metric.update(preds.clone().detach().cpu().numpy(), labels.clone().detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)
            cur_metric = metric.cur_metric()
            print('step {}/{}, loss : {}, {}'.format(step+1, dataset_sizes[phase] // batch_size, loss.item(), cur_metric), flush=True)
            sys.stdout.write("\033[F")

        epoch_loss = running_loss / dataset_sizes[phase]
        print('{} execution time {} (sec)'.format(phase, time.time() - start))
        epoch_metric = metric.metric()
        epoch_result = '{} Epoch {}, loss {}, metric {}'.format(phase, epoch+1, epoch_loss, epoch_metric)
        print(epoch_result)
        if phase != 'train':
            eval_history.append(epoch_result)

        writer.add_scalar("{}/{}".format(phase, 'Loss'), epoch_loss, epoch)
        writer.add_scalar("{}/{}".format(phase, 'Precision'), epoch_metric['precision'], epoch)
        writer.add_scalar("{}/{}".format(phase, 'Recall'), epoch_metric['recall'], epoch)
        writer.add_scalar("{}/{}".format(phase, 'F1'), epoch_metric['f1'], epoch)
        writer.add_scalar("{}/{}".format(phase, 'FPR'), epoch_metric['fpr'], epoch)

        del metric

writer.flush()
writer.close()
# 하이퍼파라미터 저장
with open(os.path.join(train_path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)

# 실험결과 저장
with open(os.path.join(train_path, 'result.txt'), 'w') as f:
    f.write(str(eval_history))

# 모델 저장하기
PATH = "state_dict_model.pt"
torch.save(model.state_dict(), os.path.join(train_path, PATH))

bot = ExamAlarmBot()
bot.send_msg('torch {} train is done, result : {}'.format(train_name, epoch_result[-1]))

# # 불러오기
# model = Net()
# model.load_state_dict(torch.load(PATH))
# model.eval()