from prefect import task, Parameter, Flow
from prefect.executors import LocalDaskExecutor

from data_processing.data_load import tfrecord_writer


@task
def write_tfrecords(images_path, destination, size):
    tfrecord_writer(images_path, destination, size)


def build_flow():
    with Flow("TFRecord Writer") as flow:
        images = Parameter("images_path")
        destination = Parameter("destination")
        size = Parameter("size")
        write_tfrecords(images, destination, size)

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor(schedules="threads")
    flow.register(project_name="cyclegan-cat")


if __name__ == "__main__":
    main()

