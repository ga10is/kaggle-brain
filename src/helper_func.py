import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold

from .common.logger import get_logger


def train_valid_split_v1(df, valid_ratio):
    df_trn, df_val = train_test_split(df,
                                      stratify=df['any'],
                                      test_size=valid_ratio,
                                      random_state=2019,
                                      shuffle=True)
    return df_trn, df_val


def train_valid_split_v2(df, valid_ich_ratio):
    n_all = (df['any'] == 1).sum()
    n_samples = int(n_all * valid_ich_ratio)
    get_logger().info('sample %d from %d for validation data' % (n_samples, n_all))

    g = df.groupby('any')['Image']
    selected = [np.random.choice(g.get_group(i).tolist(), n_samples, replace=False)
                for i in range(2)]
    selected = np.concatenate(selected, axis=0)
    np.random.shuffle(selected)

    is_val = df['Image'].isin(selected)
    df_val = df[is_val]
    df_trn = df[~is_val]

    # shuffle
    df_val = df_val.sample(frac=1, random_state=2019)
    df_trn = df_trn.sample(frac=1, random_state=2019)

    get_logger().info('train size: %d ratio(1) %f' %
                      (len(df_trn), df_trn['any'].mean()))
    get_logger().info('valid size: %d ratio(1) %f' %
                      (len(df_val), df_val['any'].mean()))
    return df_trn, df_val


def train_valid_split_v3(df, n_splits):
    get_logger().info('group split: %d' % n_splits)
    fold = GroupKFold(n_splits=n_splits)
    fold_iter = fold.split(X=df, y=df['any'], groups=df['patient_id'])
    trn_idx, val_idx = next(fold_iter)

    df_trn = df.iloc[trn_idx]
    df_val = df.iloc[val_idx]

    get_logger().info('train size: %d ratio(1) %f' %
                      (len(df_trn), df_trn['any'].mean()))
    get_logger().info('valid size: %d ratio(1) %f' %
                      (len(df_val), df_val['any'].mean()))

    return df_trn, df_val
