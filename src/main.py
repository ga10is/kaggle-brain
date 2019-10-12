import pandas as pd

from .common.logger import create_logger
from . import config
from .tvp import train, predict
from .helper_func import train_valid_split_v3


def save_split_data():
    df = pd.read_csv('./data/df_train3.csv')
    print(df.head())
    df_trn, df_val = train_valid_split_v3(df, n_splits=10)
    print(df_trn.head())
    print(df_val.head())

    df_trn.to_csv('df_trn.csv', index=False)
    df_val.to_csv('df_val.csv', index=False)


def transform_df():
    df = pd.read_csv('./data/stage1_1_train_with_metadata.csv')
    df[['ID', 'Image', 'Diagnosis']] = df['ID'].str.split('_', expand=True)
    df = df[['Image', 'Diagnosis', 'Label']]
    df.drop_duplicates(inplace=True)
    df = df.pivot(index='Image', columns='Diagnosis',
                  values='Label').reset_index()
    df['Image'] = 'ID_' + df['Image']
    print(df.head())

    df.to_csv('df_train3.csv', index=False)


if __name__ == '__main__':
    create_logger('log/brain.log')

    # train()
    # predict()
    save_split_data()
