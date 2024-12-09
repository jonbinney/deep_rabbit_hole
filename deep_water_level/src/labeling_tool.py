"""
GUI Tool for labeling water level data

Hotkeys:
  l - move forward one image
  h - move backward one image
  w - move forward one day
  b - move backward one day
  i - interpolate depths for selected image or region
  ctrl+s - save new annotations as new file
  ctrl+(left click and drag) - select region
"""

import argparse
import json
from matplotlib import pyplot as plt
import matplotlib.dates
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time

from data import WaterDataset


def parse_image_filename(image_filename: Path):
    camera, unix_time_str = re.match(r"(.+)-(.+).jpg", image_filename).groups()
    utc_dt = pd.to_datetime(int(unix_time_str), unit="s", utc=True)
    return camera, utc_dt


def load_raw_data(raw_data_dir: Path, camera_id: str = None):
    image_paths = []
    timestamps = []
    for file_path in raw_data_dir.rglob("*"):
        if not file_path.is_file():
            continue
        this_camera_id, utc_dt = parse_image_filename(file_path.parts[-1])
        if camera_id != this_camera_id:
            continue
        image_paths.append(file_path)
        timestamps.append(utc_dt)
    return pd.DataFrame({"timestamp": timestamps, "raw_image_path": image_paths})


def load_annotations(dataset_dir: Path, annotations_file_name: str):
    annotations_file_path = dataset_dir / "annotations" / annotations_file_name
    images_dir_path = dataset_dir / "images"
    dataset = WaterDataset(annotations_file_path, images_dir_path)
    data = dataset.load_annotations()
    timestamps = []
    image_paths = []
    depths = []
    x0s, y0s, x1s, y1s = [], [], [], []

    for image_path, [depth, x0, y0, x1, y1] in data:
        camera_id, utc_dt = parse_image_filename(Path(image_path).parts[-1])
        timestamps.append(utc_dt)
        image_paths.append(image_path)
        depths.append(depth)
        x0s.append(x0)
        y0s.append(y0)
        x1s.append(x1)
        y1s.append(y1)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "labeled_image_path": image_paths,
            "depth": depths,
            "x0": x0s,
            "y0": y0s,
            "x1": x1s,
            "y1": y1s,
        }
    )


def save_new_annotations(data: pd.DataFrame, new_annotations_path: Path):
    """
    Save the new annotations to a new annotations file.
    """
    images = []
    annotations = []
    for _, row in data.iterrows():
        # If this image wasn't in the dataset before and we have a depth for it, add it
        # to the new annotations file
        if pd.isnull(row["labeled_image_path"]) and not np.isclose(row["depth"], 0.0):
            image_id = len(images)
            images.append({"id": image_id, "file_name": str(Path(row["raw_image_path"]))})

            annotation_id = len(annotations)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "attributes": {"depth": row["depth"]},
                }
            )

    # Save the updated annotations to the file
    with open(new_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations}, f, indent=4)


class LabelingTool:
    def __init__(self, dataset_dir: Path, raw_images_dir: Path, new_annotations_path: Path, camera_id: str = None):
        annotations_dataframe = load_annotations(dataset_dir, args.annotations_file)
        raw_images_dataframe = load_raw_data(raw_images_dir, camera_id)
        self.data = pd.merge(raw_images_dataframe, annotations_dataframe, on="timestamp", how="outer", sort=True)
        self.new_annotations_path = new_annotations_path

        self.selection = [0, 1]  # Selection of the form [start_index, end_index]
        self.ctrl_active = False

        # Set all NaN depths to 0.0 to make plotting easier
        self.data["depth"] = self.data["depth"].fillna(0.0)

        # Load all the images into memory
        self.images = []
        for _, datapoint in self.data.iterrows():
            if pd.isnull(datapoint["raw_image_path"]):
                image_filename = datapoint["labeled_image_path"]
            else:
                image_filename = datapoint["raw_image_path"]
            self.images.append(plt.imread(image_filename))

        self.figure, self.axes = plt.subplots(2, 1)
        self.image_ax, self.plot_ax = self.axes
        self.figure.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.figure.canvas.mpl_connect("pick_event", self.on_pick)

        # Disable matplotlib keyboard shortcuts that interfere with our own
        reserved_keys = set(["h", "l", "w", "b", "i", "ctrl+s"])
        for param in plt.rcParams.keys():
            if param.startswith("keymap."):
                for key in reserved_keys.intersection(plt.rcParams[param]):
                    plt.rcParams[param].remove(key)

        # Plot the depths vs time
        self.plot_artist = self.plot_ax.scatter(
            self.data["timestamp"].dt.tz_convert("America/Argentina/Buenos_Aires"),
            self.data["depth"],
            color=["b"] * len(self.data),
            s=5,
            picker=5,
        )
        self.plot_ax.set_xlabel("Datetime (Argentina TZ)")
        self.plot_ax.set_ylabel("Depth")

        self.image_artist = None
        self.selection_artist = None
        self.water_line_artist = None
        self.update_selection()

    def update_selection(self):
        """
        Display the currently selected image.
        """

        t0 = time.time()

        print(f"Updating selection to {self.selection[0]} to {self.selection[1]}")
        image_index = self.selection[0]
        stamp_utc = self.data["timestamp"].iloc[image_index]
        stamp_argentina = stamp_utc.tz_convert("America/Argentina/Buenos_Aires")
        self.image_ax.set_title(f"Image {image_index}:    {stamp_argentina}")
        if self.image_artist is None:
            self.image_artist = self.image_ax.imshow(self.images[image_index])
        else:
            self.image_artist.set_data(self.images[image_index])

        # Clear previous water line plot
        if self.water_line_artist is not None:
            self.water_line_artist.remove()
            self.water_line_artist = None

        x0, y0, x1, y1 = (
            self.data["x0"][image_index],
            self.data["y0"][image_index],
            self.data["x1"][image_index],
            self.data["y1"][image_index],
        )
        if x0 is not None:
            (self.water_line_artist,) = self.image_ax.plot([x0, x1], [y0, y1], "r-")

        t1 = time.time()

        selection_stamps = self.data["timestamp"][self.selection[0] : self.selection[1]].dt.tz_convert(
            "America/Argentina/Buenos_Aires"
        )
        selection_depths = self.data["depth"][self.selection[0] : self.selection[1]]
        if self.selection_artist is None:
            self.selection_artist = self.plot_ax.scatter(selection_stamps, selection_depths, s=100, color="r")
        else:
            # Update the selection artist
            self.selection_artist.set_offsets(np.column_stack((selection_stamps, selection_depths)))

        t2 = time.time()

        self.figure.canvas.draw()

        t3 = time.time()

        print(f"Updating image took {t1-t0} s, updating selection took {t2-t1} seconds, drawing took {t3-t2} seconds")

    def interpolate_depths(self):
        # Find all datapoints within the selected region and set their depth to an interpolation
        # of the previous and next datapoints which have depths set.
        if self.selection[0] is not None and self.selection[1] is not None:
            print("Interpolating selected region")
            previous_known_datapoint = None
            next_known_datapoint = None
            for ind in range(self.selection[0], 0, -1):
                if self.data["depth"][ind] != 0.0:
                    previous_known_datapoint = self.data.loc[ind]
                    break
            for ind in range(self.selection[1], len(self.data), 1):
                if self.data["depth"][ind] != 0.0:
                    next_known_datapoint = self.data.loc[ind]
                    break

            if previous_known_datapoint is None:
                print("No previous datapoint with depth to use for interpolation")
                return
            elif next_known_datapoint is None:
                print("No next datapoint with depth to use for interpolation")
                return

            slope = (next_known_datapoint["depth"] - previous_known_datapoint["depth"]) / (
                next_known_datapoint["timestamp"] - previous_known_datapoint["timestamp"]
            ).total_seconds()
            for ind in range(self.selection[0], self.selection[1]):
                if self.data["depth"][ind] == 0.0:
                    self.data.loc[ind, "depth"] = previous_known_datapoint["depth"] + (
                        slope
                        * (self.data.loc[ind, "timestamp"] - previous_known_datapoint["timestamp"]).total_seconds()
                    )

            # Update depths in plot
            offsets = self.plot_artist.get_offsets()
            offsets[:, 1] = self.data["depth"].to_numpy()
            self.plot_artist.set_offsets(offsets)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def on_button_press(self, event):
        if self.ctrl_active and event.button == 1:
            start_dt = matplotlib.dates.num2date(event.xdata)
            start_ind = self.data["timestamp"].searchsorted(start_dt)
            self.selection = [start_ind, start_ind + 1]
            self.update_selection()

    def on_button_release(self, event):
        # Start selection when right mouse button is pressed
        if self.ctrl_active and event.button == 1:
            print("Drawing selected region")
            end_dt = matplotlib.dates.num2date(event.xdata)
            end_ind = self.data["timestamp"].searchsorted(end_dt)
            self.selection[1] = end_ind
            self.update_selection()

    def on_key_press(self, event):
        if event.key == "control":
            self.ctrl_active = True
        elif event.key == "i":
            print("Interpolating depths for selected region")
            self.interpolate_depths()
        elif event.key == "ctrl+s":
            print(f"Saving new annotations to {self.new_annotations_path}")
            save_new_annotations(self.data, self.new_annotations_path)
        elif event.key == "l":
            # Move forward one image
            image_index = min(self.selection[0] + 1, len(self.data) - 1)
            self.selection = [image_index, image_index + 1]
            self.update_selection()
        elif event.key == "h":
            # Move back one image
            image_index = max(self.selection[0] - 1, 0)
            self.selection = [image_index, image_index + 1]
            self.update_selection()
        elif event.key == "w":
            # Move forward one day
            new_stamp = self.data.loc[self.selection[0], "timestamp"] + pd.Timedelta(1, unit="days")
            image_index = min(self.data["timestamp"].searchsorted(new_stamp, side="left"), len(self.data) - 1)
            self.selection = [image_index, image_index + 1]
            self.update_selection()
        elif event.key == "b":
            # Move back one day
            new_stamp = self.data.loc[self.selection[0], "timestamp"] - pd.Timedelta(1, unit="days")
            image_index = max(self.data["timestamp"].searchsorted(new_stamp, side="left"), 0)
            self.selection = [image_index, image_index + 1]
            self.update_selection()

    def on_key_release(self, event):
        if event.key == "control":
            self.ctrl_active = False

    def on_pick(self, event):
        # Get the x and y coordinates of the clicked location as floats
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # Convert xdata and ydata of line section from datetimes to float positions on the axis
        offsets = event.artist.get_offsets()
        xdata = matplotlib.dates.date2num(offsets[:, 0])
        ydata = matplotlib.dates.date2num(offsets[:, 1])

        # Choose the closest point to the clicked location
        image_index = event.ind[np.argmin((xdata[event.ind] - x) ** 2 + (ydata[event.ind] - y) ** 2)]
        self.selection = [image_index, image_index + 1]
        self.update_selection()

    def spin(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Deep Water Level dataset")
    parser.add_argument(
        "--dataset-dir", type=Path, default="datasets/water_test_set3", help="Path to the dataset directory"
    )
    parser.add_argument(
        "--annotations-file", type=str, default="filtered.csv", help="File name of the JSON file containing annotations"
    )
    parser.add_argument(
        "--new-annotations-path",
        type=Path,
        default=Path("/tmp/new_annotations.json"),
        help="Where to store (just the new) annotations",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default="raw_data/pilenew/images",
        help="Path to directory containing the raw images",
    )
    parser.add_argument("--camera-id", type=str, default="0", help="Camera name")
    args = parser.parse_args()

    app = LabelingTool(args.dataset_dir, args.raw_data_dir, args.new_annotations_path, args.camera_id)
    app.spin()
