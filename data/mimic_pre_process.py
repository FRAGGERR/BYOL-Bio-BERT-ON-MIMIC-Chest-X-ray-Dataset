#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# count-total-images-text
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from constants import *
from utils import extract_mimic_text
import re

extract_text = False
np.random.seed(42)


# In[ ]:


# # Path to the main directory
# main_directory = "./files"

# def count_files(main_directory):
#     main_directory =main_directory
#     # Counters for .png and .txt files
#     png_count = 0
#     txt_count = 0

#     # Calculate total number of files for the progress bar
#     total_files = sum([len(files) for r, d, files in os.walk(main_directory)])

#     # Walk through the directory with a progress bar
#     with tqdm(total=total_files, desc="Processing files") as pbar:
#         for root, dirs, files in os.walk(main_directory):
#             for file in files:
#                 if file.endswith(".png"):
#                     png_count += 1
#                 elif file.endswith(".txt"):
#                     txt_count += 1
#                 pbar.update(1)
#     return png_count,txt_count


# In[ ]:


def pre_process_image_data():
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id","study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
    # Only keep frontal images
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]

    # Remove duplicates from chexpert_df based on subject_id and study_id
    metadata_df=metadata_df.drop_duplicates(subset=['subject_id', 'study_id'])

    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # TODO: merge validate and test into test.
    split_df["split"] = split_df["split"].apply(
        lambda x: "valid" if x == "validate" or x == "test" else x)

    # Merge split_df with metadata_df on dicom_id
    merged_df = pd.merge(split_df, metadata_df[['dicom_id', 'ViewPosition']], on='dicom_id', how='inner')

    # Filter only frontal X-rays (PA and AP views)
    frontal_df = merged_df[merged_df['ViewPosition'].isin(['PA', 'AP'])]
    frontal_df.head(), frontal_df.shape

    split_counts = frontal_df['split'].value_counts()
    print(split_counts)

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[["subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    # Merge frontal_df with chexpert_df on 'subject_id' and 'study_id'
    master_df = pd.merge(frontal_df, chexpert_df, on=['subject_id', 'study_id'], how='inner')

    n = len(master_df)
    master_data = master_df.values
    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"

    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.png" % (root_dir, str(row[2])[:2], str(row[2]), str(row[1]), str(row[0]))
        path_list.append(file_path)

    master_df.insert(loc=0, column="Path", value=path_list)


    master_df.drop(["dicom_id", "subject_id", "study_id"], axis=1, inplace=True)

    train_df = master_df.loc[master_df["split"] == "train"]
    test_df = master_df.loc[master_df["split"] == "valid"]
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
    test_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)


# In[ ]:


def pre_process_text_data():
    if extract_text:
        extract_mimic_text()
        
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id","study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
    # Only keep frontal images
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]
    
    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # TODO: merge validate and test into test.
    split_df["split"] = split_df["split"].apply(
        lambda x: "valid" if x == "validate" or x == "test" else x)
    
    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")

    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)

    n = len(master_df)
    master_data = master_df.values

    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"
    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.png" % (root_dir, str(row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
        path_list.append(file_path)

    master_df.insert(loc=0, column="Path", value=path_list)
    
    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[["subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    # Create labeled data df
    labeled_data_df = pd.merge(master_df, chexpert_df, on=[ "subject_id", "study_id"], how="inner")

    labeled_data_df.drop(["dicom_id", "subject_id", "study_id",
                          "impression", "findings"], axis=1, inplace=True)

    train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]

    TRAIN_CSV = MIMIC_CXR_DATA_DIR/'train_multiple_sids.csv'

    train_df.to_csv(TRAIN_CSV, index=False)

    valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
    TEST_CSV = MIMIC_CXR_DATA_DIR/'test_multiple_sids.csv'
    valid_df.to_csv(TEST_CSV, index=False)


    # master_df.drop(["dicom_id", "subject_id", "study_id"],
    #                axis=1, inplace=True)

    # Fill nan in text
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)


# In[ ]:


pre_process_image_data()
pre_process_text_data()



