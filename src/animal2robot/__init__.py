from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("animal2robot")
except PackageNotFoundError:
    __version__ = "0.0.0"
