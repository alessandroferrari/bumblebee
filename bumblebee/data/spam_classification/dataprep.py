#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import urllib.request
import pandas as pd
import zipfile
import os
from pathlib import Path
import sys

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
URL2 = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
ZIP_PATH = "datasets/sms_spam_collection.zip"
EXTRACTED_PATH = "bumblebee/datasets/sms_spam_collection"
DATA_FILE_PATH = "SMSSpamCollection.tsv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dataprep for spam classification.")
    parser.add_argument(
        "--train_split",
        "-t",
        default=0.7,
        type=float,
        help="How to split training set. Number included between 0.6 and 0.8. The remanining will be split between validation and testing.")
    parser.add_argument(
        "--val_split",
        "-v",
        default=0.1,
        type=float,
        help="How to split validation set. Number included between 0.1 and 0.3. The remanining will be testing.")
    parser.add_argument(
        "--balance_dataset", "-b",
        action="store_true",
        help="Downsample non spam datasets to balance label classes.")
    args = parser.parse_args()
    total_split = args.train_split + args.val_split
    if total_split > 0.9:
        print("ERROR in arguments via CLI: train_split and val_split sum should be lower or equal than 0.9.")
        sys.exit()
    return args


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def download_and_unzip_spam_data(
        url,
        zip_path,
        extracted_path,
        data_file_path):
    if os.path.exists(data_file_path):
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    print("Downloading SMS Spam Collection dataset.")
    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):

    print("Dataset is unbalanced:")
    print(df["Label"].value_counts())
    print("Balancing by downsampling ham samples.")
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    print(balanced_df["Label"].value_counts())
    return balanced_df


def random_split(df, train_frac, validation_frac):
    if (train_frac + validation_frac) > 0.9:
        raise Exception(
            f"Train split + Validation split {train_frac + validation_frac} > 0.9. Impossible to have at least 0.1 of the dataset for testing.")
    testing_frac = 1.0 - train_frac - validation_frac
    print(
        f"Splitting training / validation / testing set with ratios {train_frac:.4f} / {validation_frac:.4f} / {testing_frac:.4f}.")
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def dataprep(train_split, val_split, balance_classes=True):
    root_path = get_project_root()
    print(f"Project root path: {root_path}")
    zip_path = os.path.join(root_path, ZIP_PATH)
    extracted_path = os.path.join(root_path, EXTRACTED_PATH)
    data_file_path = os.path.join(extracted_path, DATA_FILE_PATH)

    # Download the dataset
    try:
        download_and_unzip_spam_data(
            URL, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")

        download_and_unzip_spam_data(
            URL2, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(
        data_file_path,
        sep="\t",
        header=None,
        names=[
            "Label",
            "Text"])

    if balance_classes:
        balanced_df = create_balanced_dataset(df)
    else:
        print("Skipping dataset classes balancing. Classes can result skewed.")
        balanced_df = df

    NON_SPAM = 0
    SPAM = 1
    balanced_df["Label"] = balanced_df["Label"].map(
        {"ham": NON_SPAM, "spam": SPAM})

    train_df, validation_df, test_df = random_split(
        balanced_df, train_split, val_split)
    # Test size is implied to be 0.2 as the remainder

    BASEDIR = os.path.join(root_path, "datasets", "spam_classification")
    if not os.path.exists(BASEDIR):
        os.makedirs(BASEDIR)
    train_path = os.path.join(BASEDIR, "train.csv")
    train_df.to_csv(train_path, index=None)
    validation_path = os.path.join(BASEDIR, "validation.csv")
    validation_df.to_csv(validation_path, index=None)
    test_path = os.path.join(BASEDIR, "test.csv")
    test_df.to_csv(test_path, index=None)


if __name__ == "__main__":
    args = parse_args()
    dataprep(args.train_split, args.val_split, args.balance_dataset)
