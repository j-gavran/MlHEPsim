import json
import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm


def url_download(url, data_dir, chunk_size=1024):
    """Downloads file from url to data_dir.

    Parameters
    ----------
    url : str
        URL of file to download.
    data_dir : str
        Downloaded in this directory (needs to exist).
    chunk_size : int, optional
        Chunk size for downloading, by default 1024

    References
    [1] - https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    Returns
    -------
    str
        File name of downloaded file.

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname = data_dir + url.split("/")[-1]

    if Path(fname).is_file() is not True:
        logging.warning(f"started downloading from {url} ...")

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with open(fname, "wb") as file, tqdm(
            desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    else:
        logging.warning("already downloaded!")

    return fname


def load_dataset_variables(file_dir) -> dict[str, str | list[str]]:
    json_path = file_dir + "/variables.json"

    with open(json_path, "r") as j:
        contents = json.loads(j.read())
    return contents
