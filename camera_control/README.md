### Installation

Create a python virtual environment. You must use the `--system-site-packages`
option, since gstreamer's python api can only be installed with apt, not pip.
```
virtualenv .venv --system-site-packages
```

Install GStreamer and it's python interface.
```
sudo apt install python3-gst-1.0 gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0
```

### Configuration

If you want to capture live images from a camera using gstreamer, you must specify the
`rtsp` URI for the camera. To do this, create a file called `camera_control.yml` in
`~/.config` with the following content:
```yaml
{
  camera_address: ...,
  camera_rtsp_port: ...,
  camera_rtsp_path: ...,
  camera_onvif_port: ...,
  camera_username: ...,
  camera_password: ...,
}
```

### Running the App

Use one of the two configuration in this repo's launch.json. One captures live images from
the camera; the other loads a video file from disk.
