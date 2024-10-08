### Installation

Install the following dependencies from apt:
- GStreamer (optional): Used to save camera video to disk.
- jq: used to parse the config file fromthe shell script that starts our gstreamear pipeline.

```
sudo apt install gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0
```

### Configuration

If you want to capture live images from a camera using gstreamer, you must specify the
`rtsp` URI for the camera. To do this, create a file called `camera_control.yml` in
`~/.config/` with the following content:

```yaml
{
  "camera_address": ...,
  "camera_rtsp_port": ...,
  "camera_rtsp_path": ...,
  "camera_onvif_port": ...,
  "camera_username": ...,
  "camera_password": ...,
}
```

### (optional) Start Gstreamer to Save Video Stream to Disk

The app will process images from the camera but not save them. If you want to save them,
start the gstreamer pipeline:

```
./camera_control/scripts/gstreamer_pipeline
```

### Running the App

Use one of the two configuration in this repo's launch.json. One captures live images from
the camera; the other loads a video file from disk.
