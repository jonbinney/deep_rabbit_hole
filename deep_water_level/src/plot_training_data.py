import argparse
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
import pytz
import re

from data import WaterDataset

def plot_training_data(dataset_dir: str, annotations_file: str):
    annotations_file_path = Path(dataset_dir) / 'annotations' / annotations_file
    images_dir_path = Path(dataset_dir) / 'images'
    dataset = WaterDataset(annotations_file_path, images_dir_path)
    data = dataset.load_annotations()
    argentina_tz = pytz.timezone('America/Argentina/Buenos_Aires')
    decimal_hours = []
    depths = []
    for image_path, [depth] in data:
        print(image_path, depth)
        unix_time_str = re.match(r'.+-(.+).jpg', Path(image_path).name).groups()[0]
        dt = datetime.datetime.utcfromtimestamp(int(unix_time_str))
        utc_dt = pytz.utc.localize(dt)
        argentina_dt = utc_dt.astimezone(argentina_tz)
        decimal_hour = argentina_dt.hour + argentina_dt.minute / 60 + argentina_dt.second / 3600
        decimal_hours.append(decimal_hour)
        depths.append(depth)
        print(argentina_dt)
    plt.scatter(decimal_hours, depths)
    plt.title('Depth vs Hour of data collection (Argentina TZ)')
    plt.xlabel('Decimal hour')
    plt.ylabel('Depth')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--dataset_dir', type=str, default='datasets/water_2024_10_19_set1', help='Path to the dataset directory')
    parser.add_argument('--annotations_file', type=str, default='manual_annotations.json', help='File name of the JSON file containing annotations')
    args = parser.parse_args()

    plot_training_data(args.dataset_dir, args.annotations_file)
