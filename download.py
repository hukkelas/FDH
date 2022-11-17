import webdataset as wds
import json
import click
from tqdm import tqdm
from pathlib import Path
from hashlib import md5
import os
from urllib.request import urlopen, Request

md5sums = {
    "cc-by-2": "4277732684f20518635f243e6a18d378",
    "cc-by-nc-2": "f87ca81af4e8a250709e840f29f80dca",
    "cc-by-sa-2": "7f86cef7e86386505510026e17d72241",
    "cc-by-nc-sa-2": "81ac9e42dc61b3ab3f73240aa5f19109",
}

image_urls = {
    "cc-by-nc-sa-2": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/1d10ad76-c287-430f-a8ed-3eeee332843b66dc1e33-115e-4414-9ded-d933ec7cd0463ea89ca4-396a-4625-b943-9d0b577c7bf2",
    "cc-by-2": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/ac8fc219-e2fa-4b61-a650-2a653b9960257670465a-3b37-4fb4-8861-c28030b9461004bb7226-000c-471e-9a41-a6591f0da642",
    "cc-by-sa-2": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/e3213fdf-8ead-4245-853b-6d0d2f9286abf14e26ca-d1a8-40af-99de-fad54f4fa8d0350fbcfa-a26c-45f7-9dfa-1d934176bc58",
    "cc-by-nc-2": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/a7f71c51-c1b4-423a-b9af-df944f8eb341f13dbab0-5a64-407c-9eb7-b6a46b0f5e657f747d12-72c0-4945-af0d-57da785e4c79",
}

fdh_embedding_path = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/1bfde92d-bcde-4142-9345-1bc1188644fcd4f3b9fd-78af-46ad-aac2-03c5ebec4046125a8ff6-6fd3-4271-9e6b-ecfeeddc04da"
fdh_embedding_md5sum = "a8d793f6c7e625e19d1ad97c763f2fbe"
fdh_metainfo_path = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/dbd67368-45c4-4c61-b517-e1493fc2ad7347dc09ca-e0e3-4394-8bac-b539e9321ba91670fa6f-3584-4000-9acd-d3f403844784"
fdh_metainfo_md5sum = "220b8e5f495a6bd60aed267f66da1c5a"
embed_map_path = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/026d3286-3c38-4352-aa87-6dab3d0dc78d972254b8-8f61-493a-8241-11da7f8532b32e6296b9-74a9-4fa1-84e5-e773c9e46620"
embed_map_md5sum = "456ef247f0f1f0f393d1b10310d571a4"


def md5_for_file(f, block_size=2**20):
    md5 = md5()
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.digest()


def download_url_to_file(url, dst, hash_prefix=None): # Modified from: https://pytorch.org/docs/1.12/_modules/torch/hub.html#download_url_to_file
    file_size = None
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    f = open(dst, "wb")
    md5sum = md5()
    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            if hash_prefix is not None:
                md5sum.update(buffer)
            pbar.update(len(buffer))

    f.close()
    digest = md5sum.hexdigest()
    if digest[:len(hash_prefix)] != hash_prefix:
        raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
    f.close()


def download(url, target_path: Path, md5sum):
    target_path.parent.mkdir(exist_ok=True, parents=True)
    if target_path.is_file():
        print("File already downloaded:", target_path)
        return
#        target_path.unlink()
    print("Downloading:", url)
    download_url_to_file(url, str(target_path), md5sum)


class TarIterator:
    def __init__(self, tar_path: Path) -> None:
        self.pipeline = iter(wds.DataPipeline(
            wds.SimpleShardList([str(tar_path)]),
            wds.tarfile_to_samples()
        ))
        self.fetch_next()

    def fetch_next(self):
        try:
            self.next = next(self.pipeline)
        except StopIteration:
            self.next = None

    def has_next(self):
        return self.next is not None

    def has(self, key):
        return self.has_next() and key == self.next["__key__"]
    
    def is_empty(self):
        return self.next is None


@click.command()
@click.argument("target_path")
def main(target_path):
    target_path = Path(target_path)
    download(embed_map_path, target_path.joinpath("embed_map.torch"), embed_map_md5sum)
    download(fdh_embedding_path, target_path.joinpath("embeddings.tar"), fdh_embedding_md5sum)
    download(fdh_metainfo_path, target_path.joinpath("metadata.json"), fdh_metainfo_md5sum)

    for image_license, image_url in image_urls.items():
        print("Downloading images with license:", image_license)
        download(image_url, target_path.joinpath(image_license + ".tar"), md5sums[image_license])

    # Extract dataset
    embedding_reader = wds.DataPipeline(
            wds.SimpleShardList([str(target_path.joinpath("embeddings.tar"))]),
            wds.tarfile_to_samples()
    )
    image_readers = dict()
    for license in image_urls.keys():
        image_readers[license] = TarIterator(target_path.joinpath(license + ".tar"))

    target_path.joinpath("train").mkdir(exist_ok=True, parents=True)
    target_path.joinpath("val").mkdir(exist_ok=True, parents=True)
    shard_writers = dict(
        train=wds.ShardWriter(str(target_path.joinpath("train", "out-%06d.tar")), maxsize=200*1024*1024, maxcount=3000), # maxsize 200MBs
        val=wds.ShardWriter(str(target_path.joinpath("val", "out-%06d.tar")), maxsize=200*1024*1024, maxcount=3000) # maxsize 200MBs
    )
    with open(target_path.joinpath("metadata.json")) as fp:
        metainfo = json.load(fp)

    for sample in tqdm(embedding_reader, desc="Writing samples to shards.", total=1870743):
        key = sample["__key__"]
        split = metainfo[key]["cateogry"]

        writer = shard_writers[split]
        n_readers_has_key = 0
        for im_reader in image_readers.values():
            if im_reader.has(key):
                sample.update(im_reader.next)
                im_reader.fetch_next()
                n_readers_has_key += 1
        assert n_readers_has_key == 1, n_readers_has_key
        writer.write(sample)
    for reader in image_readers.values():
        assert reader.is_empty()
    print("The dataset is sucessfully extracted. You can safely delete the downloaded tars:")
    print("\t rm -r ", *[str(target_path.joinpath(p)) for p in ["embeddings.tar"] + [f"{ln}.tar" for ln in image_urls.keys()]])


main()