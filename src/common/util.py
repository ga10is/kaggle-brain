import sys
from sklearn.model_selection import KFold
import pandas as pd


def debug_deco(func):
    def wrapper(*args, **kwargs):
        print('--start--')
        from IPython.core.debugger import Pdb
        Pdb().set_trace()
        ret = func(*args, **kwargs)
        print('--end--')
        return ret
    return wrapper


def debug_trace(func):
    def wrapper(*args, **kwargs):
        print('--start-- %s' % sys._getframe().f_code.co_name)
        ret = func(*args, **kwargs)
        print('--end--')
        return ret
    return wrapper


def split_train_valid(df, y, n_splits):
    # fold = StratifiedKFold(n_splits=n_splits, random_state=2019, shuffle=True)
    fold = KFold(n_splits=n_splits, random_state=2019, shuffle=True)
    # ignore y if KFold
    iter_fold = fold.split(df, y)
    idx_train, idx_valid = next(iter_fold)
    df_train = df.iloc[idx_train]
    df_valid = df.iloc[idx_valid]

    return df_train, df_valid


def str_stats(data):
    """
    print statistics

    Parameters
    ----------
    data: 1-d numpy.ndarray
    """
    stats = pd.DataFrame(pd.Series(data).describe()).transpose()
    return str(stats)
