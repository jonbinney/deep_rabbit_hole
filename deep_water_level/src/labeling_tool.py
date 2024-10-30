import argparse
from matplotlib import pyplot as plt
import matplotlib.dates
import numpy as np
import pandas as pd
from pathlib import Path
import re

from data import WaterDataset

def parse_image_filename(image_filename: Path):
    camera, unix_time_str = re.match(r'(.+)-(.+).jpg', image_filename).groups()
    utc_dt = pd.to_datetime(int(unix_time_str), unit='s', utc=True)
    return camera, utc_dt

def load_raw_data(raw_data_dir: str, camera_id: str = None):
    # Loop through all image files in the raw data directory
    search_path = Path(raw_data_dir)
    image_paths = []
    timestamps = []

    for file_path in search_path.rglob('*'):
        if not file_path.is_file():
            continue
        this_camera_id, utc_dt = parse_image_filename(file_path.parts[-1])
        if camera_id is not None and camera_id != this_camera_id:
            continue
        image_paths.append(file_path)
        timestamps.append(utc_dt)
    return pd.DataFrame({ 'timestamp': timestamps, 'raw_image_path': image_paths})

def load_annotations(dataset_dir: str, annotations_file: str, raw_data_dir: str):
    annotations_file_path = Path(dataset_dir) / 'annotations' / annotations_file
    images_dir_path = Path(dataset_dir) / 'images'
    dataset = WaterDataset(annotations_file_path, images_dir_path)
    data = dataset.load_annotations()
    timestamps = []
    image_paths = []
    depths = []
    for image_path, [depth] in data:
        camera_id, utc_dt = parse_image_filename(Path(image_path).parts[-1])
        timestamps.append(utc_dt)
        image_paths.append(image_path)
        depths.append(depth)
    return pd.DataFrame({'timestamp': timestamps, 'labeled_image_path': image_paths, 'depth': depths})

class LabelingTool:
    def __init__(self, raw_images: pd.DataFrame, annotations: pd.DataFrame):
        self.data = pd.merge(raw_images, annotations, on='timestamp', how='outer')
        self.displayed_image = 0 # Index of displayed image in data dataframe
        self.selection = [None, None] # Selection of the form [start, end]
        self.ctrl_active = False
        self.drawn_selection = None

        # Set all NaN depths to 0.0 to make plotting easier
        self.data['depth'] = self.data['depth'].fillna(0.0)
                        
        self.fig, self.axes = plt.subplots(2, 1)
        self.image_ax, self.plot_ax = self.axes
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        # Plot the depths vs time
        self.plot_artist, = self.plot_ax.plot(
            self.data["timestamp"].dt.tz_convert('America/Argentina/Buenos_Aires'),
            self.data["depth"],
            'bo',
            picker=5)
        self.plot_ax.set_xlabel('Datetime (Argentina TZ)')
        self.plot_ax.set_ylabel('Depth')

        self.draw()
    
    def draw(self):
        print(self.selection)
        self.image_ax.figure.canvas.draw()

    def on_button_press(self, event):
        print(f'click: button={event.button}, x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}')

        # Start selection when right mouse button is pressed
        if self.ctrl_active and event.button == 1:
            if self.drawn_selection is not None:
                self.drawn_selection.remove()
            self.selection = [event.xdata, None]
        
        self.draw()

    def on_button_release(self, event):
        print(f'click: button={event.button}, x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}')

        # Start selection when right mouse button is pressed
        if self.ctrl_active and event.button == 1:
            self.selection[1] = event.xdata

        if self.selection[0] is not None and self.selection[1] is not None:
            print("Drawing selected region")
            start = self.selection[0]
            end = self.selection[1]
            self.drawn_selection = self.plot_ax.axvspan(start, end, facecolor='green', alpha=0.2)
        
        self.draw()
    
    def on_key_press(self, event):
        if event.key == "control":
            self.ctrl_active = True
        elif event.key == 'i':
            # Find all datapoints within the selected region and set their depth to an interpolation
            # of the previous and next datapoints which have depths set.
            if self.selection[0] is not None and self.selection[1] is not None:
                print("Interpolating selected region")
                start_dt = matplotlib.dates.num2date(self.selection[0])
                end_dt = matplotlib.dates.num2date(self.selection[1])
                selection_start_ind = self.data['timestamp'].searchsorted(start_dt)
                selection_end_ind = self.data['timestamp'].searchsorted(end_dt)
                previous_known_datapoint = None
                next_known_datapoint = None
                for ind in range(selection_start_ind, 0, -1):
                    if self.data['depth'][ind] != 0.0:
                        previous_known_datapoint = self.data.loc[ind]
                        break
                for ind in range(selection_end_ind, len(self.data), 1):
                    if self.data['depth'][ind] != 0.0:
                        next_known_datapoint = self.data.loc[ind]
                        break

                if  previous_known_datapoint is None:
                    print('No previous datapoint with depth to use for interpolation')
                    return
                elif  next_known_datapoint is None:
                    print('No next datapoint with depth to use for interpolation')
                    return

                slope = ((next_known_datapoint['depth'] - previous_known_datapoint['depth'])
                    / (next_known_datapoint['timestamp'] - previous_known_datapoint['timestamp']).total_seconds())
                for ind in range(selection_start_ind, selection_end_ind):
                    if self.data['depth'][ind] == 0.0:
                        self.data.loc[ind, 'depth'] = previous_known_datapoint['depth'] + (slope * 
                            (self.data.loc[ind, 'timestamp'] - previous_known_datapoint['timestamp']).total_seconds())
                
                self.plot_artist.set_ydata(self.data['depth'])
                self.draw()

    def on_key_release(self, event):
        if event.key == "control":
            self.ctrl_active = False

    def on_pick(self, event):
        # Get the x and y coordinates of the clicked location as floats
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # Convert xdata and ydata of line section from datetimes to float positions on the axis
        line = event.artist
        xdata, ydata = line.get_data()
        xdata = matplotlib.dates.date2num(xdata)
        ydata = matplotlib.dates.date2num(ydata)

        # Choose the closeset point to the clicked location
        self.displayed_image = event.ind[np.argmin((xdata[event.ind] - x)**2 + (ydata[event.ind] - y)**2)]

        # Display the image
        if self.data['raw_image_path'].isnull().iloc[self.displayed_image]:
            image_filename = self.data['labeled_image_path'].iloc[self.displayed_image]
        else:
            image_filename = self.data['raw_image_path'].iloc[self.displayed_image]
        image = plt.imread(image_filename)
        self.image_ax.imshow(image)
        self.draw()
    
    def spin(self):
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Deep Water Level dataset')
    parser.add_argument('--dataset-dir', type=str, default='datasets/water_2024_10_19_set1', help='Path to the dataset directory')
    parser.add_argument('--annotations-file', type=str, default='manual_annotations.json', help='File name of the JSON file containing annotations')
    parser.add_argument('--raw-data-dir', type=str, default='raw_data/pilenew/images', help='Path to directory containing the raw images')
    parser.add_argument('--camera-id', type=str, default='0', help='Camera name')
    args = parser.parse_args()

    raw_dataframe = load_raw_data(args.raw_data_dir, args.camera_id)
    annotations = load_annotations(args.dataset_dir, args.annotations_file, args.raw_data_dir)

    app = LabelingTool(raw_dataframe, annotations)
    app.spin()
