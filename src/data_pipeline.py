import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(_project_root, "data", "withLabel")
test_file = "140207_1.xlsx"


def load_excel_files(data_dir=data_dir):
    """Load all Excel files. Joins two-row headers. Ref to the paper and its preprocessing."""

    paths = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.xlsx') and not f.startswith('~$')  # Remove '~$' files if they are created
    ])

    df_list = []
    for path in paths:
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sheet_name=0, index_col=0, header=[0, 1])
        df.columns = df.columns.map(''.join)
        df = df.rename_axis('time').reset_index()
        col_names = list(df.columns)
        col_names[-1] = 'label'
        df.columns = col_names
        df_list.append(df)

    return df_list, paths


def avg_out_point1(df):
    """Replace point-1 CO2 with point-2 block average. Ref to the paper and its preprocessing."""

    label_list = list(df['label'])
    conc_list = list(df['AT400(CO2 %)'])

    # Loop 1: block averages for label==2
    p, num = 0, 0
    sampling2_avg = []
    i = 0
    while i < len(label_list):
        if label_list[i] == 2:
            p += conc_list[i]
            num += 1
            i += 1
        else:
            if p == 0 and num == 0:
                i += 1
            else:
                sampling2_avg.append(p / num)
                i += 1
                num, p = 0, 0

    # Loop 2: overwrite label==1 with point-2 avg
    i, k = 0, -1
    while i < len(label_list) - 1:
        if label_list[i] == 1:
            conc_list[i] = sampling2_avg[k]
            i += 1
        else:
            if label_list[i + 1] == 1:
                k += 1
            i += 1

    df['AT400(CO2 %)'] = pd.Series(data=conc_list, index=df.index)

    return df


def column_separator(df):
    """Create 6 interpolated CO2 columns (1_sampling ... 6_sampling).
    For each point: keep CO2 where label matches, NaN elsewhere, then interpolate.
    Ref to the paper and its preprocessing.
    """

    for i in range(1, 7):
        new_con = []
        for j in range(df.shape[0]):
            if df.iloc[j]['label'] == i:
                new_con.append(df.iloc[j]['AT400(CO2 %)'])
            else:
                new_con.append(np.nan)
        col_name = f"{i}_sampling"
        df[col_name] = pd.Series(data=new_con, index=df.index)
        df[col_name] = df[col_name].interpolate(method="linear")

    return df


def prepare_data(data_dir=data_dir, test_file=test_file):
    """Load, preprocess, and normalize all data files. Ref to the paper and its preprocessing."""

    df_list, paths = load_excel_files(data_dir)

    for i in range(len(df_list)):
        df_list[i] = df_list[i].set_index('time')
        df_list[i] = avg_out_point1(df_list[i])

    # Split by filename
    train_dfs, test_dfs = [], []
    for df, path in zip(df_list, paths):
        if os.path.basename(path) == test_file:
            test_dfs.append(df)
        else:
            train_dfs.append(df)

    # Fit two scalers on training data only: general_scaler, conc_scaler for CO2
    train_features = np.concatenate(
        [df.iloc[:, :-1].values for df in train_dfs], axis=0
    )
    train_conc = np.concatenate(
        [df['AT400(CO2 %)'].values for df in train_dfs], axis=0
    )

    general_scaler = MinMaxScaler()
    conc_scaler = MinMaxScaler()
    general_scaler.fit(train_features)
    conc_scaler.fit(train_conc.reshape(-1, 1))

    # Normalize and create sampling columns
    all_dfs = train_dfs + test_dfs
    for df in all_dfs:
        df.iloc[:, :-1] = general_scaler.transform(df.iloc[:, :-1].values)
        column_separator(df)
        df.fillna(0, inplace=True)

    # 90 sensor feature names (exclude label + sampling cols)
    exclude = {'label'} | {f"{i}_sampling" for i in range(1, 7)}
    feature_names = sorted([c for c in all_dfs[0].columns if c not in exclude])

    return train_dfs, test_dfs, general_scaler, conc_scaler, feature_names


class CO2Dataset(Dataset):
    """Sliding-window the dataset. x: (input_window, 96), y: (forecast_window, 6). Ref to the paper and its preprocessing"""

    def __init__(self, df_list, input_window, forecast_window, feature_names):
        self.samples = []
        self.targets = []

        sampling_cols = [f"{i}_sampling" for i in range(1, 7)]

        for df in df_list:
            features = df[feature_names].values  # (n, 90), n: rows in each df

            # One-hot encode label (1-6)
            labels = df['label'].values
            onehot = np.zeros((len(labels), 6))
            for k in range(len(labels)):
                point = int(labels[k])
                if 1 <= point <= 6:
                    onehot[k, point - 1] = 1

            full_features = np.hstack([features, onehot])  # (n, 96)
            targets = df[sampling_cols].values  # (n, 6)

            # Create sample sequences via sliding windows
            n_seq = len(df) - input_window - forecast_window + 1
            for j in range(n_seq):
                x = full_features[j : j + input_window] # past records
                y = targets[j + input_window : j + input_window + forecast_window] # forcasting window
                self.samples.append(x.astype(np.float32))
                self.targets.append(y.astype(np.float32))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], self.targets[idx]


def create_dataloaders(train_dfs, test_dfs, feature_names, input_window=17, 
                       forecast_window=1, batch_size=32):
    """Create train, val, test DataLoaders. Val = every 5th sample (80/20). Ref to the paper and its preprocessing."""

    full_train = CO2Dataset(train_dfs, input_window, forecast_window, feature_names)
    test_dataset = CO2Dataset(test_dfs, input_window, forecast_window, feature_names)

    # Validation: every 5th sample 
    n = len(full_train)
    val_indices = list(range(0, n, 5))
    train_indices = [i for i in range(n) if i not in val_indices]

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
