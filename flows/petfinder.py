import os
import petpy
import urllib.error
import urllib.request

from pathlib import Path
from operator import add
from functools import reduce
from prefect.executors import LocalDaskExecutor
from prefect import task, Parameter, Flow, mapped, unmapped, flatten


@task
def get_photo_urls(api, breed, pages):
    df = api.animals(animal_type="cat",
            breed=breed,
            results_per_page=100,
            pages=pages,
            return_df=True)
    medium_urls = df["photos"].map(lambda x: [y["large"] for y in x])
    photo_urls = reduce(add, medium_urls.to_list())

    return photo_urls


@task(nout=2)
def generate_save_paths(urls, destination, breed):
    breed_dir = Path(destination, breed)
    breed_dir.mkdir(parents=True, exist_ok=True)
    save_paths = [breed_dir.joinpath(f"{i:05}.png") for i, _ in enumerate(urls)]
    return save_paths


@task
def download_photo(url, save_path):
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
        save_paths = generate_save_paths(mapped(work_list), unmapped(destination), mapped(breeds))
        download_photo(flatten(mapped(work_list)), flatten(mapped(save_paths)))

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor(schedules="threads")
    flow.register(project_name="cyclegan-cat")


if __name__ == "__main__":
    main()

