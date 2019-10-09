from sklearn.model_selection import train_test_split


def train_valid_split_v1(df, valid_ratio):
    df_trn, df_val = train_test_split(df,
                                      stratify=df['any'],
                                      test_size=valid_ratio,
                                      random_state=2019,
                                      shuffle=True)
    return df_trn, df_val
