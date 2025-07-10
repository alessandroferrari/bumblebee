#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import os
import urllib
import urllib.request


URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
FILE_RELATIVE_PATH = "datasets/instruct_sft/instruction-data.json"


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def download_and_load_instruct_sft_data():

    file_path = os.path.join(get_project_root(), FILE_RELATIVE_PATH)
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.exists(file_path):
        with urllib.request.urlopen(URL) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    download_and_load_instruct_sft_data()
