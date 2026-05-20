from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class CameraDataProtocol(Protocol):
    """
    Canonical interface for camera data consumed by the optimization pipeline.

    Every camera source loader (VIPER, COLMAP, etc.) must produce an object
    that exposes these attributes.  External processes handle the translation
    from raw formats; this protocol is the middle-layer specification.
    """

    cam_R: torch.Tensor   # (T, 3, 3)  world-to-camera rotation
    cam_t: torch.Tensor   # (T, 3)     world-to-camera translation
    intrins: torch.Tensor # (T, 4)     [fx, fy, cx, cy]
    is_static: bool

    def world2cam(self):
        """Forward transform: world → camera."""
        return self.cam_R, self.cam_t

    def cam2world(self):
        """Inverse transform: camera → world."""
        R = self.cam_R.transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, self.cam_t)
        return R, t

    def as_dict(self) -> dict:
        """Canonical dict consumed by ``CameraParams.set_cameras()``."""
        return {
            "cam_R": self.cam_R,
            "cam_t": self.cam_t,
            "intrins": self.intrins,
            "static": self.is_static,
        }
