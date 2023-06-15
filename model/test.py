import numpy as np
from transformers import DistilBertTokenizer, BertTokenizer
from CFG import CFG
from train import build_loaders
from Model import CLIPModel
import torch
from tqdm.autonotebook import tqdm
import torch.nn.functional as F


def get_image_embedding(test, model_path):
    # tokenizer_d = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    # tokenizer_b = BertTokenizer.from_pretrained(CFG.label_tokenizer)
    # test_loader = build_loaders(test, tokenizer_d, tokenizer_b, mode="valid")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    test_loader = build_loaders(test, tokenizer, mode="valid")

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    text_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=batch['input_id'].to(CFG.device), attention_mask=batch['attention_m'].to(CFG.device))

            # image_embeddings = model.fc_i(image_features)
            # text_embeddings = model.fc_t(text_features)

            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)
            i_t_embeddings = torch.add(image_embeddings, text_embeddings)

            # i_t_embeddings = torch.cat((image_features, text_features), dim=1)

            # i_t_embeddings = model.cat_projection(i_t_embeddings)

            # i_t_embeddings = model.fc_it(i_t_embeddings)
            # i_t_embeddings = model.gelu(i_t_embeddings)
            # i_t_embeddings = model.fc_it2(i_t_embeddings)

            # i_t_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)
            # i_t_embeddings = model.cat_pojrection(i_t_embeddings)
            # text_image_embeddings.append(image_embeddings)
            text_image_embeddings.append(i_t_embeddings)
    return model, torch.cat(text_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames):
    # tokenizer = BertTokenizer.from_pretrained(CFG.label_tokenizer)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer(query, padding=True)

    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        # text_features = model.text_encoder(input_ids=batch["input_id"], attention_mask=batch["attention_m"])
        text_features = model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # text_features = model.label_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # text_embeddings = model.text_projection(text_features)
        text_embeddings = model.fc_l(text_features)
        # text_embeddings = model.layer_norm(text_embeddings)
        # text_embeddings = model.dropout(text_embeddings)
    # image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    # text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

    # logit_scale = (torch.ones([]) * -0.2).exp()
        logit_scale = model.logit_scale.exp()
    logits = logit_scale * text_embeddings @ image_embeddings.T
    probs = logits.T.softmax(dim=-1).cpu().numpy()
    print(probs)
    return probs


