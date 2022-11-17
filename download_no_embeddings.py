import webdataset as wds
from download import (
    md5sums,
    image_urls,
    fdh_metainfo_path,
    fdh_metainfo_md5sum,
    download,
    TarIterator,
)
import json
import tqdm
import click
from pathlib import Path


@click.command()
@click.argument("target_path")
def main(target_path):
    target_path = Path(target_path)
    download(fdh_metainfo_path, target_path.joinpath("metadata.json"), fdh_metainfo_md5sum)
    for image_license, image_url in image_urls.items():
        print("Downloading images with license:", image_license)
        download(image_url, target_path.joinpath(image_license + ".tar"), md5sums[image_license])

    image_readers = dict()
    for license in image_urls.keys():
        image_readers[license] = TarIterator(target_path.joinpath(license + ".tar"))

    target_path.joinpath("train").mkdir(exist_ok=True, parents=True)
    target_path.joinpath("val").mkdir(exist_ok=True, parents=True)
    shard_writers = dict(
        train=wds.ShardWriter(str(target_path.joinpath("train", "out-%06d.tar")), maxsize=200 * 1024 * 1024, maxcount=3000),  # maxsize 200MBs
        val=wds.ShardWriter(str(target_path.joinpath("val", "out-%06d.tar")), maxsize=200 * 1024 * 1024, maxcount=3000),  # maxsize 200MBs
    )
    with open(target_path.joinpath("metadata.json")) as fp:
        metainfo = json.load(fp)
    pbar = tqdm.tqdm(desc="Writing samples to shards.", total=1870743)
    while True:
        # Find reader with lowest key
        cur_reader = None
        for reader in image_readers:
            if not reader.has_next():
                continue
            if cur_reader is None:
                cur_reader = reader
                continue
            if int(reader.next["__key__"]) < int(cur_reader.next["__key__"]):
                cur_reader = reader
        if cur_reader is None:
            break
        sample = cur_reader.next
        key = sample["__key__"]
        split = metainfo[key]["cateogry"]
        shard_writers[split].write(sample)
        cur_reader.fetch_next()
        pbar.update(1)
    for reader in image_readers.values():
        assert reader.is_empty()
    for writer in shard_writers.values():
        writer.close()


main()
