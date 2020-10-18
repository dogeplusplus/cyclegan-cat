import argparse

from data_processing.data_load import tfrecord_writer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create tfrecords for training.")
    parser.add_argument("--images", type=str, help="Path to images.")
    parser.add_argument("--dest", type=str, help="Path to store the tfrecords.")
    parser.add_argument("--width", type=int, help="Width of image.")

    args = parser.parse_args()
    return args

def main(args):
    tfrecord_writer(args.images, args.dest, image_size=args.width)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
