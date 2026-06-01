"""
Utility modules — logging, Ray orchestration, MIDI I/O.

Usage:
    >>> from hlstm_framework.utils import setup_logger
    >>> logger = setup_logger("hlstm")
"""

from hlstm_framework.utils.logging_utils import setup_logger, Timer
from hlstm_framework.utils.ray_utils import is_ray_available, RayTrainWrapper
from hlstm_framework.utils.midi_io import write_midi, write_wav, combine_midi

__all__ = [
    "setup_logger",
    "Timer",
    "is_ray_available",
    "RayTrainWrapper",
    "write_midi",
    "write_wav",
    "combine_midi",
]
