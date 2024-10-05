import datetime
import gi
import numpy as np
import pathlib

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class GStreamerCameraCapture:
    """
    Class to read frames from a GStreamer pipeline using the appsink element.

    We don't use the opencv VideoCapture class with the GStreamer backend because
    it isn't included in the default opencv installation.
    """
    def __init__(self, camera_uri):
        Gst.init(None)
        home_dir = pathlib.Path.home()
        datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%Z")
        pipeline_str = " " .join([
            f'rtspsrc location={camera_uri} latency=30 ! rtph265depay ! h265parse ! tee name=t ',
            ' t. ! queue ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink ',
            f' t. ! queue ! splitmuxsink max-size-time=3600000000000 location={home_dir}/Downloads/rabbits_{datetime_str}_%02d.mp4',
            ])
        print('Creating gstreamer pipeline: {}'.format(pipeline_str))
        self.pipeline = Gst.parse_launch(pipeline_str)
            
        self.appsink = self.pipeline.get_by_name("appsink")
        if not self.appsink:
            raise ValueError("Pipeline must contain an element named 'appsink'")

        self.pipeline.set_state(Gst.State.PLAYING)

    def read_next_frame(self):
        sample = self.appsink.emit("pull-sample")
        if not sample:
            raise ValueError("Failed to read frame from pipeline")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        new_image = np.ndarray(
            (
                caps.get_structure(0).get_value("height"),
                caps.get_structure(0).get_value("width"),
                3,
            ),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8,
        )
        return new_image

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)
