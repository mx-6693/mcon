import numpy as np
import pandas as pd
from train import train_CLIP
from test import get_image_embedding, find_matches
from sklearn.metrics import classification_report, accuracy_score
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# def append_label_as_text(dataframe):
#     dataframe.loc[dataframe['sentiment'] == 1, 'text'] = dataframe.loc[dataframe['sentiment'] == 1].apply(
#             lambda row: row[1] + ' [SEP] a meme with happiness emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 2, 'text'] = dataframe.loc[dataframe['sentiment'] == 2].apply(
#             lambda row: row[1] + ' [SEP] a meme with love emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 3, 'text'] = dataframe.loc[dataframe['sentiment'] == 3].apply(
#             lambda row: row[1] + ' [SEP] a meme with angry emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 4, 'text'] = dataframe.loc[dataframe['sentiment'] == 4].apply(
#             lambda row: row[1] + ' [SEP] a meme with sorrow emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 5, 'text'] = dataframe.loc[dataframe['sentiment'] == 5].apply(
#             lambda row: row[1] + ' [SEP] a meme with fear emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 6, 'text'] = dataframe.loc[dataframe['sentiment'] == 6].apply(
#             lambda row: row[1] + ' [SEP] a meme with hate emotion', axis=1)
#     dataframe.loc[dataframe['sentiment'] == 7, 'text'] = dataframe.loc[dataframe['sentiment'] == 7].apply(
#             lambda row: row[1] + ' [SEP] a meme with surprise emotion', axis=1)


def change_label_as_text(dataframe, label_map):
    dataframe['label_to_text'] = dataframe['sentiment'].map(label_map)


def change_intention_as_text(dataframe, intention_map):
    dataframe['intention_to_text'] = dataframe['intention'].map(intention_map)


def main(dir_path, train_model=0, task=7):
    label_map = {
        1: ' [CLS] a meme with happiness emotion',
        2: ' [CLS] a meme with love emotion',
        3: ' [CLS] a meme with anger emotion',
        4: ' [CLS] a meme with sorrow emotion',
        5: ' [CLS] a meme with fear emotion',
        6: ' [CLS] a meme with hate emotion',
        7: ' [CLS] a meme with surprise emotion'
    }
    intention_map = {
        1: 'a meme with interactive intention',
        2: 'a meme with expressive intention',
        3: 'a meme with entertaining intention',
        4: 'a meme with offensive intention',
        5: 'a meme with other intention'
    }
    if task == 7 and train_model == 1:
        train = pd.read_csv(dir_path + '/train.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        # append_label_as_text(train)
        train['text'] = train['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        train['target_text'] = train['target_text'].apply(lambda x: x.replace(';', ' '))
        train['source_text'] = train['source_text'].apply(lambda x: x.replace(';', ' '))
        train['text'] = ' [CLS] ' + train['text'] + ' [SEP] ' + train['source_text'] + ' [SEP] ' + train['target_text']
        # print(train['text'])
        train = train.sample(frac=1.0)
        train = train.reset_index(drop=True)
        change_label_as_text(train, label_map)

        valid = pd.read_csv(dir_path + '/val.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        # append_label_as_text(valid)
        valid['text'] = valid['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        valid['target_text'] = valid['target_text'].apply(lambda x: x.replace(';', ' '))
        valid['source_text'] = valid['source_text'].apply(lambda x: x.replace(';', ' '))
        valid['text'] = ' [CLS] ' + valid['text'] + ' [SEP] ' + valid['source_text'] + ' [SEP] ' + valid['target_text']

        valid = valid.sample(frac=1.0)
        valid = valid.reset_index(drop=True)
        change_label_as_text(valid, label_map)

        print("training the model...")
        train_CLIP(train, valid)
    elif task == 5 and train_model == 1:
        train = pd.read_csv(dir_path + '/train.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        # append_label_as_text(train)
        train['text'] = train['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        # train['target_text'] = train['target_text'].apply(lambda x: x.replace(';', ' '))
        # train['source_text'] = train['source_text'].apply(lambda x: x.replace(';', ' '))
        # train['text'] = train['text'] + ' [SEP] ' + train['source_text'] + ' [SEP] ' + train['target_text']
        # print(train['text'])
        train = train.sample(frac=1.0)
        train = train.reset_index(drop=True)
        change_intention_as_text(train, intention_map)

        valid = pd.read_csv(dir_path + '/val.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        # append_label_as_text(valid)
        valid['text'] = valid['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        # valid['target_text'] = valid['target_text'].apply(lambda x: x.replace(';', ' '))
        # valid['source_text'] = valid['source_text'].apply(lambda x: x.replace(';', ' '))
        # valid['text'] = valid['text'] + ' [SEP] ' + valid['source_text'] + ' [SEP] ' + valid['target_text']

        valid = valid.sample(frac=1.0)
        valid = valid.reset_index(drop=True)
        change_intention_as_text(valid, intention_map)

        print("training the model...")
        train_CLIP(train, valid)
    elif task == 7 and train_model == 0:
        test = pd.read_csv(dir_path + '/test.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        test['text'] = test['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        test['target_text'] = test['target_text'].apply(lambda x: x.replace(';', ' '))
        test['source_text'] = test['source_text'].apply(lambda x: x.replace(';', ' '))
        test['text'] = ' [CLS] ' + test['text'] + ' [SEP] ' + test['source_text'] + ' [SEP] ' + test['target_text']

        test = test.sample(frac=1.0)
        test = test.reset_index(drop=True)
        change_label_as_text(test, label_map)
        model, image_embeddings = get_image_embedding(test, "cls_add_pro_fc_loss.pt")

        probs = []
        y_pred = []

        for index, row in test.iterrows():
            # query = [row['text'] + " [SEP] a meme with happiness emotion",
            #          row['text'] + " [SEP] a meme with love emotion",
            #          row['text'] + " [SEP] a meme with angry emotion",
            #          row['text'] + " [SEP] a meme with sorrow emotion",
            #          row['text'] + " [SEP] a meme with fear emotion",
            #          row['text'] + " [SEP] a meme with hate emotion",
            #          row['text'] + " [SEP] a meme with surprise emotion"
            # ]
            query = ["a meme with happiness emotion",
                     "a meme with love emotion",
                     "a meme with angry emotion",
                     "a meme with sorrow emotion",
                     "a meme with fear emotion",
                     "a meme with hate emotion",
                     "a meme with surprise emotion"]
            prob = find_matches(model, image_embeddings[index].unsqueeze(0), query, image_filenames=test['file_name'].values)
            probs.append(prob)
            y_pred.append(np.argmax(prob) + 1)
        print('y_pred: ', y_pred)

        y_true = test['sentiment'].values.tolist()
        print(y_true)
        target_names = ['happiness', 'love', 'anger', 'sorrow', 'fear', 'hate', 'surprise']
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4), '\n')
        # print("AC:", accuracy_score(y_true, y_pred))
    elif task == 5 and train_model == 0:
        test = pd.read_csv(dir_path + '/test.csv', header='infer', keep_default_na=False, encoding='ISO-8859-1')
        test['text'] = test['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

        # test['target_text'] = test['target_text'].apply(lambda x: x.replace(';', ' '))
        # test['source_text'] = test['source_text'].apply(lambda x: x.replace(';', ' '))
        # test['text'] = test['text'] + ' [SEP] ' + test['source_text'] + ' [SEP] ' + test['target_text']

        test = test.sample(frac=1.0)
        test = test.reset_index(drop=True)
        change_intention_as_text(test, intention_map)
        model, image_embeddings = get_image_embedding(test, "5_add_pro_fc_3.pt")

        probs = []
        y_pred = []

        for index, row in test.iterrows():
            # query = [row['text'] + " [SEP] a meme with happiness emotion",
            #          row['text'] + " [SEP] a meme with love emotion",
            #          row['text'] + " [SEP] a meme with angry emotion",
            #          row['text'] + " [SEP] a meme with sorrow emotion",
            #          row['text'] + " [SEP] a meme with fear emotion",
            #          row['text'] + " [SEP] a meme with hate emotion",
            #          row['text'] + " [SEP] a meme with surprise emotion"
            # ]
            query = ["a meme with interactive intention",
                     "a meme with expressive intention",
                     "a meme with entertaining intention",
                     "a meme with offensive intention",
                     "a meme with other intention"
                     ]
            prob = find_matches(model, image_embeddings[index].unsqueeze(0), query,
                                image_filenames=test['file_name'].values)
            probs.append(prob)
            y_pred.append(np.argmax(prob) + 1)
        print('y_pred: ', y_pred)

        y_true = test['intention'].values.tolist()
        print(y_true)
        target_names = ['interactive', 'expressive', 'entertaining', 'offensive', 'other']
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4), '\n')
        # print("AC:", accuracy_score(y_true, y_pred))
dir_path = "../data"
# train = pd.read_csv(dir_path+'\\train.csv', header='infer', keep_default_na=False)
# for i in range(len(train['text'])):
#     train['text'][i] = train['text'][i].replace('\n', '').replace('\r', '')
#     train['text'][i] = ' '.join(train['text'][i].split())
# train.to_csv("D:\\memes_clip\\data\\changed_train.csv")


main(dir_path, 0, 5)

