"""Provides enums and helper functions related to colors/palettes/etc.
"""
from enum import unique, Enum


@unique
class Colors(Enum):
    """Provides hex values for various colors"""

    PASTEL_BLUE = '#7AA9CF'
    TULIP_TREE = '#EBB13E'
    CUSTOM_GREEN = '#41B3A3'
    CRAIL = '#A65B50'
    DOVE_GRAY = '#7A7A7A'
    CUSTOM_PURPLE = '#C38D9E'
    CADMIUM_ORANGE = '#F28E2B'

    MARINER = '#4E79A7'
    SANDSTONE = '#F4CC70'
    AVOCADO = '#258039'
    TOMATO = '#CF3721'
    MIST = '#90AFC5'
    LAVENDER = '#C396E8'
    TREE_POPPY = '#FD9126'

    CERULEAN = '#1EB1ED'
    YELLOW_PEPPER = '#F5BE41'
    PIGMENT_GREEN = '#1AAF54'
    POPPY = '#FF420E'
    LOBLOLLY_GRAY = '#B4B7B9'
    LAVENDER_2 = '#6C648B'
    FLAMINGO = '#FC641F'

    FOREST = '#1E434C'
    GOLD = '#C99E10'
    SUNFLOWER = '#3F681C'
    CRIMSON = '#8D230F'
    GRANITE = '#B7B8B6'
    VIVID_VIOLET = '#932791'
    PETAL = '#F98866'

    BLACK_SHADOW = '#2A3132'
    SKY = '#375E97'
    MEDIUM_SEA_GREEN = '#37B57F'
    RED_CLAY = '#A43820'


GOOD = Colors.AVOCADO.value
BAD = Colors.TOMATO.value
GOOD_BAD = (GOOD, BAD)
WARNING = Colors.TULIP_TREE.value
ERROR = BAD

GRAY = Colors.LOBLOLLY_GRAY.value
