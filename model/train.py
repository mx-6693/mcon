from Dataset import CLIPDataset
from Dataset import get_transforms
import torch
from CFG import CFG
from utils import AvgMeter, get_lr
from torch.utils import data
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer, BertTokenizer
from Model import CLIPModel
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloaders

# def build_loaders(dataframe, tokenizer_d, tokenizer_b, mode):
#     transforms = get_transforms(mode=mode)
#     dataset = CLIPDataset(
#         dataframe,
#         tokenizer_d=tokenizer_d,
#         tokenizer_b=tokenizer_b,
#         transforms=transforms,
#     )
#     dataloaders = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=CFG.batch_size,
#         num_workers=CFG.num_workers,
#         shuffle=True if mode == "train" else False,
#     )
#     return dataloaders


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption" and k != "intention_to_text"}
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption" and k != "label_to_text"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    # torch.save(model.state_dict(),"train.pt")
    # print("Saved train model")
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        # batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption" and k != "intention_to_text"}
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption" and k != "label_to_text"}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def train_CLIP(train, validation):
    # tokenizer_d = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    # tokenizer_b = BertTokenizer.from_pretrained(CFG.label_tokenizer)
    # train_loader = build_loaders(train, tokenizer_d, tokenizer_b, mode="train")
    # valid_loader = build_loaders(validation, tokenizer_d, tokenizer_b, mode="valid")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train, tokenizer, mode="train")
    valid_loader = build_loaders(validation, tokenizer, mode="valid")

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        # {"params": model.label_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": model.fc_l.parameters(), "lr": CFG.head_lr},
        # {"params": model.dropout.parameters(), "lr": CFG.head_lr},
        # {"params": model.text_projection.parameters(), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
        # {"params": model.cat_projection.parameters(), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        # {"params": model.image_projection.parameters(), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        # {"params": model.fc_it.parameters(), "lr": CFG.head_lr},
        # {"params": model.gelu.parameters(), "lr": CFG.head_lr},
        # {"params": model.fc_it2.parameters(), "lr": CFG.head_lr}
        # {"params": model.fc_i.parameters(), "lr": CFG.head_lr},
        # {"params": model.fc_t.parameters(), "lr": CFG.head_lr}
        # {"params": model.layer_norm.parameters(), "lr": CFG.head_lr},
        {"params": model.logit_scale, "lr": CFG.head_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    cnt = 0
    best_loss = float('inf')
    train_l = []
    val_l = []
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        train_l.append(train_loss)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            val_l.append(valid_loss.avg)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "cls_add_pro_fc_loss.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

    # x1 = range(0,50)
    #x2 = range(0,50)
    #y1 = train_l
    #y2 = val_l
    #plt.plot(x1, y1)
    #plt.plot(x1,y2,color='red')
    #plt.xlabel('epoch')
    #plt.ylabel('Loss')
    #plt.show()

