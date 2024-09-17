from onvif import ONVIFCamera
from pathlib import Path
import sys
import time
import yaml

class OnvifCameraController:
    def __init__(self, host, port, user, password):
        """
        Simple interface for controlling a camera using the ONVIF protocol.
        """
        self.camera = ONVIFCamera(host, port, user, password)
        self.media_service = self.camera.create_media_service()
        self.ptz_service = self.camera.create_ptz_service()
        self.media_profile = self.media_service.GetProfiles()[0]
        self.ptz_configuration = self.ptz_service.GetConfigurationOptions({'ConfigurationToken': self.media_profile.PTZConfiguration.token})

    def stop(self):
        """
        Works with Jennov PS6007.
        """
        self.ptz_service.Stop({'ProfileToken': self.media_profile.token})

    def continuous_move(self, x, y, z):
        """
        Works for pan and tilt but not zoom with Jennov PS6007.
        """
        request = self.ptz_service.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': x, 'y': y},
            'Zoom': {'x': z}
        }
        self.ptz_service.ContinuousMove(request)


    def relative_move(self, pan, tilt, zoom):
        """
        Seems to be the same as a "ContinuousMove" for pan and tilt on Jennov PS6007.
        """
        request = self.ptz_service.create_type('RelativeMove')
        request.ProfileToken = self.media_profile.token
        request.Translation = {
            'PanTilt': {'x': pan, 'y': tilt},
            'Zoom': {'x': zoom}
        }
        self.ptz_service.RelativeMove(request)

    def goto_preset(self, preset_token):
        """
        Works with Jennov PS6007
        """
        request = self.ptz_service.create_type('GotoPreset')
        request.ProfileToken = self.media_profile.token
        request.PresetToken = preset_token
        self.ptz_service.GotoPreset(request)

    def is_movement_finished(self):
        """
        Doesn't seem to work with Jennov PS6007, at least when moving to presets.
        """
        request = self.ptz_service.create_type('GetStatus')
        request.ProfileToken = self.media_profile.token
        status = self.ptz_service.GetStatus(request)
        return status.MoveStatus == 'IDLE'

if __name__ == "__main__":
    # Load camera configuration from a YAML file.
    config_file_path = Path.home() / ".config/camera_control.yml"
    try:
        with open(config_file_path) as yaml_file:
            camera_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"No config file found at {config_file_path}.")
        sys.exit(-1)

    cam = OnvifCameraController(
        camera_config["camera_address"],
        camera_config["camera_onvif_port"],
        camera_config["camera_username"],
        camera_config["camera_password"])


    for ii in range(3):
        print("Moving to preset 1")
        cam.goto_preset("1")
        time.sleep(10.0)


        print("Moving right and down")
        cam.continuous_move(1, 1, 0)
        time.sleep(3.0)

        cam.stop()
        time.sleep(2.0)


        print("Zooming in")
        cam.relative_move(0, 0, 1)
        time.sleep(5.0)

        print('Stopping')
        cam.stop()
        time.sleep(2.0)
