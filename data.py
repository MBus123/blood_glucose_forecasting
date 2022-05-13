import torch
import pandas as pd
import numpy as np

import random
import os

from torch.utils.data import Dataset

from collections import defaultdict
from functools import reduce



class OhioData(Dataset):
    """
    The representation of the OHIO dataset prepared for pytorch (train, validation, test). 
    Missing values will be interpolated using cubic splines and all values are scaled between 0 and 1. 

    Attributes
    ----------
    mode : str
        train, validation or test
    train_val_ration : float
        defines the split between train and validation data
    h : int
        the prediction horizon
    T : int
        the amount of trimesteps a model is allowed to look back
    folders : list(str)
        which data for training/testing to include
    drop_cols : list(str)
        which columns to exclude for prediction
    is_transformer : bool
        a transformer requires different input than the other models
    patient_id : int
        include only a single patient (only used for testing)

    Methods
    -------
    preprocess_data(df):
        Adds positional encoding and interpolates missing data 
    
    """

    def __init__(self, mode="train", train_val_ratio=0.8, h=6, T=24, folders=["Ohio2018", "Ohio2020"], drop_cols=[], is_transformer=False, patient_id=None):
        """
        Loads the data from file and prepares them for training/inference
        """
        super(OhioData, self).__init__()
        self.mode = mode
        self.is_transformer = is_transformer
        self.h = h
        self.T = T
        train_or_test = "train" if mode in ["train", "validation"] else "test"
        self.use_val_data = False
        self.drop_cols = drop_cols
        self.df_list = []

        # get scale values from train data. Even when preparing the test set, one cannot scale by them because the model was trained by scale of the train values
        tmp_df = []
        for folder_name in folders: 
            for r, d, f in os.walk(os.path.join("Ohio Data", folder_name, "train")):
                for file in f:
                    df = pd.read_csv(os.path.join(r, file))
                    split = round(df.shape[0] * train_val_ratio)
                    tmp_df.append(df[:split])
        tmp_df = pd.concat(tmp_df, ignore_index=True)
        self.scale_max = {key: tmp_df[key].max() if tmp_df[key].isna().sum() != len(tmp_df[key]) else 0 for key in tmp_df.columns}
        self.scale_min = {key: tmp_df[key].min() if tmp_df[key].isna().sum() != len(tmp_df[key]) else 0 for key in tmp_df.columns}
        del tmp_df, self.scale_max["5minute_intervals_timestamp"], self.scale_min["5minute_intervals_timestamp"]

        for folder_name in folders:
            for r, d, f in os.walk(os.path.join("Ohio Data", folder_name, train_or_test)):
                for file in f:
                    if patient_id != None and file != f"{patient_id}-ws-testing_processed.csv":
                        continue
                    df = self.preprocess_data(pd.read_csv(os.path.join(r, file)))
                    split = round(df.shape[0] * train_val_ratio)
                    if mode == "train":
                        df = df[:split]
                    elif mode == "validation":
                        df = df[split:]
                        self.first_example = df[:h + T]

                    # split df into sub_dfs where only non missing values are included
                    y = np.diff(df["missing_cbg"].to_numpy())
                    np.where(y == 1), np.where(y == -1)
                    z1 = np.where(y == 1)[0]
                    z2 = np.where(y == -1)[0]
                    # print(df["missing_cbg"])
                    if df["missing_cbg"].iloc[0] == 0:
                        z2 = np.append(np.array([-1]), z2)
                    if df["missing_cbg"].iloc[-1] == 0:
                        z1 = np.append(z1, np.array([df.shape[0]]))
                    z1, z2
                    for i, j in zip(z2, z1):
                        sub_df = df.iloc[i + 1:j + 1]
                        self.df_list.append(sub_df)
                    # self.df_list.append(df)
        
        for _, df in enumerate(self.df_list):
            for key in self.scale_min.keys():
                if self.scale_max[key] == 0:
                    continue
                # TODO: Fix scaling
                df[key] /= self.scale_max[key] 
            df.fillna(-1, inplace=True)

        if mode == "validation":
            for key in self.scale_min.keys():
                if self.scale_max[key] == 0:
                    continue
                self.first_example[key] /= self.scale_max[key] 
            self.first_example.fillna(-1, inplace=True)

        self.length = [ x.shape[0] for x in self.df_list ]
        self.max = [ x - self.h - self.T + 1 for x in self.length]
            

        # amount_samples = math.floor(df.shape[0] / T) - h # results into ignoring the last h * T data points (worst case)
        # amount_samples = df.shape[0] - T - h + 1
        
    def preprocess_data(self, df):
        """
        Adds positional encoding and interpolates missing feature values. 
        """
        # for key in self.scale_max.keys(): 
        #     self.scale_max[key] = df[key].max() if df[key].max() > self.scale_max[key] and len(df[key]) != df[key].isna().sum() else self.scale_max[key]
        #     self.scale_min[key] = df[key].min() if df[key].min() < self.scale_min[key] and len(df[key]) != df[key].isna().sum() else self.scale_min[key]
        for key in ["gsr", "basal", "hr"]:
            df[key] = df[key].interpolate("cubic")
        df["5minute_intervals_timestamp"] -= df["5minute_intervals_timestamp"].min()
        df["5minute_intervals_timestamp"] /= df["5minute_intervals_timestamp"].max()

        # temporal encoding
        df["time_sin"] = np.sin(df["5minute_intervals_timestamp"] * 2 * np.pi / 288)
        df["time_cos"] = np.cos(df["5minute_intervals_timestamp"] * 2 * np.pi / 288)
        # df = df.to_numpy().flatten()

        return df

    def __len__(self):
        """
        Returns the amount of different forecasting pairs.
        """
        max = [ x - self.h - self.T - 1 for x in self.length]
        return sum(max)

    def __getitem__(self, item):
        """
        Returns the pair of data used for training/inference.

        Parameters
        ----------
        item : int
            The index of the to returned pair 
        """
        max = [ x - self.h - self.T - 1 for x in self.length]
        df = None
        for i in range(len(max)):
            if item <= max[i]: 
                df = self.df_list[i] 
                break
            item -= max[i]

        # TODO: Drop Rows, parameter list
        # df_input = df_input.drop
        df_input = df[item:item + self.T] if not self.is_transformer else df[item:item + self.T + self.h]
        if self.is_transformer:
            pd.options.mode.chained_assignment = None  # default='warn'
            df_input["categorial_col"] = 0
        df_label = df[item + self.T: item + self.T + self.h]["cbg"]
        df_mask = df[item + self.T: item + self.T + self.h ]
        df_mask = 1 - df_mask["missing_cbg"]
        # TODO: Remove where input is missing
        for col in self.drop_cols:
            df_input = df_input.drop(col)     

        x = df_input.to_numpy(dtype=np.float32).flatten()
        y = df_label.to_numpy(dtype=np.float32).flatten()
        mask = df_mask.to_numpy(dtype=np.float32)

        return x, y, mask

    def get_first_example(self):
        df_input = self.first_example[:self.T] if not self.is_transformer else self.first_example
        if self.is_transformer:
            pd.options.mode.chained_assignment = None  # default='warn'
            df_input["categorial_col"] = 0
        df_label = self.first_example[self.T:self.T + self.h]["cbg"]
        x = df_input.to_numpy(dtype=np.float32).flatten()
        y = df_label.to_numpy(dtype=np.float32).flatten()
        inp = df_input["cbg"].to_numpy(dtype=np.float32).flatten()

        return x, y, inp


