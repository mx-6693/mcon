import torch


class CFG:
    debug = False
    image_path = "../data/Eimages"
    batch_size = 16
    num_workers = 0
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "resnet50"
    image_embeddding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    # label_tokenizer = "bert-base-uncased"
    # label_encoder_model = "bert-base-uncased"
    # cat_embedding = 2816
    max_length = 200

    pretrained = True
    trainable = True
    temperature = 1.0

    # image size
    size = 224

    #for projection head
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.2
