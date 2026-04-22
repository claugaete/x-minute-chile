from .amenities import Amenity, osm_amenity
from .origins import Origins
from .ratings import AccessibilityRatings
from .travel_time import TravelTimeMatrices
from .visualization import OverlayConfig

__all__ = [
    "Amenity",
    "osm_amenity",
    "Origins",
    "AccessibilityRatings",
    "TravelTimeMatrices",
    "OverlayConfig",
]

# default projected CRS (Chile)
projected_crs = 5361

# QuackOSM working directory (for storing cached files)
quackosm_working_directory = "files"

# path to osmconvert (for executing extract_osm_subset)
osmconvert_path = "osmconvert"

# default alpha to use when roads are shown on a plot (might be useful to match
# the colors of the plot with the colors of the legend if they are made
# separately)
alpha_when_roads_shown = 0.8
