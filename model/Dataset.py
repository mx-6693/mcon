import torch.utils.data
from CFG import CFG
import cv2
import albumentations as A
# import clip


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transforms):
        self.image_filenames = list(dataframe['file_name'])
        self.captions = dataframe['text']
        self.encoded_captions = tokenizer(
            list(dataframe['text']), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.label_to_text = dataframe['label_to_text']
        self.encoded_l_to_t = tokenizer(
            list(dataframe['label_to_text']), padding=True, truncation=True, max_length=CFG.max_length
        )

        # self.intention_to_text = dataframe['intention_to_text']
        # self.encoded_i_to_t = tokenizer(
        #     list(dataframe['intention_to_text']), padding=True, truncation=True, max_length=CFG.max_length
        # )

        # self.encoded_l_to_t = clip.tokenzize(list(dataframe['label_to_text']))

        self.transforms = transforms

    # def __init__(self, dataframe, tokenizer_d, tokenizer_b, transforms):
    #     self.image_filenames = list(dataframe['file_name'])
    #     self.captions = dataframe['text']
    #     self.encoded_captions = tokenizer_d(
    #         list(dataframe['text']), padding=True, truncation=True, max_length=CFG.max_length
    #     )
    #     self.label_to_text = dataframe['label_to_text']
    #     self.encoded_l_to_t = tokenizer_b(
    #         list(dataframe['label_to_text'])
    #     )
    #
    #     # self.encoded_l_to_t = clip.tokenzize(list(dataframe['label_to_text']))
    #
    #     self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        item['input_id'] = item.pop('input_ids')
        item['attention_m'] = item.pop('attention_mask')
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        # print(CFG.image_path+self.image_filenames[idx])
        # if image is None:
           # print("image in None")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['label_to_text'] = self.label_to_text[idx]
        # item['intention_to_text'] = self.intention_to_text[idx]

        for k, v in self.encoded_l_to_t.items():
            item[k] = torch.tensor(v[idx])

        # for k, v in self.encoded_i_to_t.items():
        #     item[k] = torch.tensor(v[idx])

        # item['encoded_l_to_t'] = self.encoded_l_to_t[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )
