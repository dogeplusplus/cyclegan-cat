import os
import petpy
import uuid
import urllib

from pathlib import Path
from operator import add
from functools import reduce
from prefect import task, Parameter, Flow, mapped, unmapped
from prefect.executors import LocalDaskExecutor
from typing import Tuple, List


@task
def get_photo_urls(api, breed, pages):
    df = api.animals(animal_type="cat",
            breed=breed,
            results_per_page=100,
            pages=pages,
            return_df=True)
    medium_urls = df["photos"].map(lambda x: [y["medium"] for y in x])
    photo_urls = reduce(add, medium_urls.to_list())

    return photo_urls


@task
def batch_download_photo(photo_urls, breed_name, destination):
    breed_folder = Path(destination, breed_name)
    breed_folder.mkdir(parents=True, exist_ok=True)
    for url in photo_urls:
        save_path = breed_folder.joinpath(f"{uuid.uuid4()}.jpg")
        urllib.request.urlretrieve(url, save_path)


def build_flow():
    key = os.getenv("PETFINDER_KEY")
    secret = os.getenv("PETFINDER_SECRET")
    pf = petpy.Petfinder(key, secret)

    with Flow("Petfinder Batch Image Downloader") as flow:

        breeds = Parameter("breeds")
        destination = Parameter("destination")
        pages = Parameter("pages")

        work_list = get_photo_urls(unmapped(pf), mapped(breeds), unmapped(pages))
        batch_download_photo(mapped(work_list), mapped(breeds), unmapped(destination))

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor()
    flow.register(project_name="cyclegan-cat")


if __name__ == "__main__":
    main()

