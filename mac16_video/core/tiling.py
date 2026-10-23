import copy
import dataclasses

from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig


def _rep(o, **kw):
    if dataclasses.is_dataclass(o):
        return dataclasses.replace(o, **kw)
    o = copy.copy(o)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def make_cfg(sp, sp_ov, tf, tf_ov):
    """Custom tiling: spatial px, spatial overlap px, temporal frames, temporal overlap."""
    base = TilingConfig.aggressive()
    s = _rep(base.spatial_config, tile_size_in_pixels=sp, tile_overlap_in_pixels=sp_ov)
    t = _rep(base.temporal_config, tile_size_in_frames=tf, tile_overlap_in_frames=tf_ov)
    return _rep(base, spatial_config=s, temporal_config=t)
