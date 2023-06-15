import torch
import numpy as np
from CFG import CFG
from torch import nn
from encoder import ImageEncoder, TextEncoder, ProjectionHead
import torch.nn.functional as F



class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embeddding,
        text_embedding=CFG.text_embedding,
        # cat_embedding=CFG.cat_embedding
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        # self.label_encoder = LabelEncoder()

        # self.bn_image = nn.BatchNorm1d(2048, momentum=0.5)
        # self.bn_text = nn.BatchNorm1d(768, momentum=0.5)

        # self.fc_it = nn.Linear(2816, 256)
        # self.gelu = nn.GELU()
        # self.fc_it2 = nn.Linear(256, 256)

        # self.fc_i = nn.Linear(2048, 256)
        # self.fc_t = nn.Linear(768, 256)

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

        # self.label_model, self.preprocess = clip.load("VIT-B/32", device=CFG.device)

        # self.text_projection = ProjectionHead(embedding_dim=text_embedding)

        self.fc_l = nn.Linear(768, 256)
        # self.dropout = nn.Dropout(p=0.15)

        # self.cat_projection = ProjectionHead(embedding_dim=cat_embedding)

        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * -0.2)

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_id"], attention_mask=batch["attention_m"]
        )

        # i_t_embedding = torch.cat((image_features, text_features), dim=1)

        # i_t_embedding = self.cat_projection(i_t_embedding)

        # i_t_embedding = self.fc_it(i_t_embedding)
        # i_t_embedding = self.gelu(i_t_embedding)
        # i_t_embedding = self.fc_it2(i_t_embedding)

        # image_features = self.bn_image(image_features)
        # text_features =self.bn_text(text_features)

        # image_embeddings = self.fc_i(image_features)
        # text_embeddings = self.fc_t(text_features)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        i_t_embedding = torch.add(image_embeddings, text_embeddings)

        label_to_text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # label_to_text_features = self.label_encoder(
        #     input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        # )

        # with torch.no_grad():
        #     label_to_text_features = self.label_model.encode_text(batch['encoded_l_to_t'])

        # label_to_text_embeddings = self.text_projection(label_to_text_features)

        label_to_text_embeddings = self.fc_l(label_to_text_features)

        # label_to_text_embeddings = self.dropout(label_to_text_embeddings)

        # i_t_embedding = self.cat_projection(i_t_embedding)

        logit_scale = self.logit_scale.exp()

        # logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        # logits_per_image = logit_scale * image_embeddings @ label_to_text_embeddings.T
        logits_per_image = logit_scale * i_t_embedding @ label_to_text_embeddings.T
        logits_per_text = logits_per_image.T

        # batch_size = CFG.batch_size
        batch_size = logits_per_image.shape[0]

        i_t_similarity = i_t_embedding @ i_t_embedding.T
        label_to_text_similarity = label_to_text_embeddings @ label_to_text_embeddings.T
        labels = F.softmax(
            (i_t_similarity + label_to_text_similarity) / 2, dim=-1
        )
        i_t_loss = cross_entropy(logits_per_image, labels, reduction="none")
        label_to_text_loss = cross_entropy(logits_per_text, labels.T, reduction="none")
        total_loss = (i_t_loss + label_to_text_loss) / 2.0

        # labels = torch.arange(batch_size, device=CFG.device).long()
        # total_loss = (
        #         F.cross_entropy(logits_per_image, labels) +
        #         F.cross_entropy(logits_per_text, labels)
        # ) / 2
        return total_loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
