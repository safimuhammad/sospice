from dataclasses import dataclass
from platformdirs import user_data_path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

import sunpy
from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.net import Fido, attrs
import sunpy_soar  # noqa: F401

from ..catalog import Catalog, FileMetadata


# set global figure fontsize parameters
plt.rcParams["axes.labelsize"] = "xx-large"
plt.rcParams["axes.titlesize"] = "xx-large"
plt.rcParams["xtick.labelsize"] = "x-large"
plt.rcParams["ytick.labelsize"] = "x-large"


def _show_or_save(fig, ax, show, save):
    """
    Show figure or save it to a file
    """
    if save is not None:
        fig.savefig(save)
    if show:
        fig.show()
        fig = None
        ax = None
    return fig, ax


@dataclass
class FovBackground:
    """
    Map intended as background for plotting SPICE FOVs

    Parameters
    ----------
    map_type: str
        Type of background map
    cat: sospice.Catalog
        SPICE observations, used to background map parameters if needed
    time: datetime.datetime, astropy.time.Time, pandas.Timestamp...
        Reference time used to find background map
    observer: astropy.coordinates.SkyCoord
        Coordinates of the observer
    """

    map_type: str = "default"
    cat: Catalog = None
    time: pd.Timestamp = None
    observer: SkyCoord = None

    map_types = {
        "default": "plot_blank_helioprojective",
        # "blank": "plot_blank_helioprojective",
        # "blank_hp": "plot_blank_helioprojective",
        "blank_helioprojective": "plot_blank_helioprojective",
        "HMI_synoptic": "plot_HMI_synoptic",
        "EUI/FSI": "plot_EUI_FSI",
    }

    def __post_init__(self):
        self.check_arguments()

    def check_arguments(self):
        """
        Check class arguments and try guessing missing parameters:
        time from cat or observer, observer from cat and time.
        """
        if self.map_type not in self.map_types:
            raise NotImplementedError(
                f"Map type not implemented, choose one of {', '.join(self.map_types)}"
            )
        if self.time is None:
            if self.cat is not None and len(self.cat) > 0:
                self.time = self.cat.mid_time()
            elif self.observer is not None:
                self.time = pd.Timestamp(self.observer.obstime.to_datetime(), tz="UTC")
            else:
                raise RuntimeError(
                    "If time is not provided, either observer or cat should be provided"
                )
        elif type(self.time) is str:
            self.time = pd.Timestamp(self.time)
        if self.observer is None:
            time_delta = self.cat["DATE-BEG"] - self.time
            i_closest = abs(time_delta).argmin()
            closest = FileMetadata(self.cat.iloc[i_closest], skip_validation=True)
            self.observer = closest.get_observer()

    def plot_HMI_synoptic(self):
        """
        Plot SDO/HMI synoptic map, intended as a background for plotting SPICE FOVs

        Return
        ------
        matplotlib.figure.Figure
            Figure
        matplotlib.axes.Axes
            Axes (with relevant projection)
        """
        carrington_rotation = int(np.floor(carrington_rotation_number(self.time)))
        url = f"http://jsoc.stanford.edu/data/hmi/synoptic/hmi.Synoptic_Mr.{carrington_rotation}.fits"
        syn_map = Map(url)
        fig = plt.figure(figsize=(19, 9.5))
        ax = fig.add_subplot(projection=syn_map)
        syn_map.plot(
            axes=ax, title=f"SDO/HMI synoptic map for CR {carrington_rotation}"
        )
        return fig, ax

    def plot_EUI_FSI(self, delta_t=pd.Timedelta(hours=1)):
        """
        Plot Solar Orbiter/EUI/FSI map, intended as a background for plotting SPICE FOVs

        Parameters
        ----------
        delta_t: pandas.Timedelta
            Half-width of time interval in which to look to EUI data

        Return
        ------
        matplotlib.figure.Figure
            Figure
        matplotlib.axes.Axes
            Axes (with relevant projection)
        """
        results_fsi = Fido.search(
            attrs.Time(self.time - delta_t, self.time + delta_t),
            attrs.soar.Product("eui-fsi174-image"),
            attrs.Level(2),
        )
        n_found = len(results_fsi[0])
        str_n_found = f"for {self.time} ± {delta_t}"
        if n_found == 0:
            raise RuntimeError("No file found " + str_n_found)
            # TODO rather revert to blank map
        else:
            print(f"{n_found} files found for {str_n_found}")
        delay = pd.Series(results_fsi[0]["Start time"]).apply(pd.Timestamp) - self.time
        i_closest = abs(delay).argmin()
        path = user_data_path(appname="sospice", ensure_exists=True)
        fsi_file = Fido.fetch(results_fsi[0][i_closest], path=path / "{file}")
        fsi_filename = fsi_file[0]
        fig = plt.figure(figsize=(10, 10))
        fsi_map = Map(fsi_filename)
        ax = fig.add_subplot(projection=fsi_map)
        fsi_map.plot(axes=ax)
        return fig, ax

    def plot_blank_helioprojective(self):
        """
        Plot Solar Orbiter/EUI/FSI map, intended as a background for plotting SPICE FOVs

        Return
        ------
        matplotlib.figure.Figure
            Figure
        matplotlib.axes.Axes
            Axes (with relevant projection)
        """
        # raise NotImplementedError("Blank map background not implemented yet")
        data = np.full((10, 10), np.nan)

        obs_heligraphic_sth = self.observer
        obstime = self.observer.obstime
        # obs_helioprojective=obs_heligraphic_sth.transform_to(frames.Helioprojective)

        skycoord = SkyCoord(
            0 * u.arcsec,
            0 * u.arcsec,
            obstime=obstime,
            observer=obs_heligraphic_sth,
            frame=frames.Helioprojective,
        )
        header = sunpy.map.make_fitswcs_header(
            data, skycoord, scale=[1000, 1000] * u.arcsec / u.pixel
        )
        blank_map = sunpy.map.Map(data, header)

        # fig = plt.figure()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(projection=blank_map)
        blank_map.plot(axes=ax)
        blank_map.draw_limb(axes=ax, color="k")
        blank_map.draw_grid(axes=ax, color="k")

        return fig, ax

    def plot_map(self, show=True, save=None, **kwargs):
        """
        Plot map, intended as a background for plotting SPICE FOVs

        Parameters
        ----------
        show: bool
            Show figure. Then returns None, None
        save: str
            File to save figure to
        kwargs: dict
            Additional parameters for the plotting functions

        Return
        ------
        matplotlib.figure.Figure
            Figure
        matplotlib.axes.Axes
            Axes (with relevant projection)
        """
        plot_function = getattr(self, self.map_types[self.map_type])
        fig, ax = plot_function(**kwargs)
        return _show_or_save(fig, ax, show, save)


def plot_fov_background(
    map_type=None, cat=None, time=None, observer=None, show=True, save=None, **kwargs
):
    """
    Plot map, intended as a background for plotting SPICE FOVs

    Parameters
    ----------
    map_type: str
        Type of background map
    cat: sospice.Catalog
        SPICE observations, used to determine background map parameters if needed
    time: datetime.datetime, astropy.time.Time, pandas.Timestamp...
        Reference time used to find background map
    observer: astropy.coordinates.SkyCoord
        Coordinates of the observer
    show: bool
        Show figure. Then returns None, None
    save: str
        File to save figure to
    kwargs: dict
        Additional parameters for the plotting functions

    Return
    ------
    matplotlib.figure.Figure
        Figure
    matplotlib.axes.Axes
        Axes (with relevant projection)
    """
    fov_background = FovBackground(
        map_type=map_type, cat=cat, time=time, observer=observer
    )
    fig, ax = fov_background.plot_map(show=False, **kwargs)
    return _show_or_save(fig, ax, show, save)


def plot_fovs_with_background(
    cat,
    map_type=None,
    time=None,
    observer=None,
    show=True,
    fig=None,
    ax=None,
    save=None,
    bg_kwargs=dict(),
    **kwargs,
):
    """
    Plot SPICE FOVs on a background map

    Parameters
    ----------
    cat: sospice.Catalog
        SPICE observations
    map_type: str
        Type of background map
    time: datetime.datetime, astropy.time.Time, pandas.Timestamp...
        Reference time used to find background map
    observer: astropy.coordinates.SkyCoord
        Coordinates of the observer
    fig: matplotlib.figure.Figure
        Figure on which to do the plot
    ax: matplotlib.axes.Axes
        Axis on which the FOVs should be plotted; this needs to have a relevant projection.
    show: bool
        Show figure. Then returns None, None
    save: str
        File to save figure to
    bg_kwargs: dict
        Keyword arguments for plot_fov_background()
    kwargs: dict
        Keyword arguments for Catalog.plot_fov()

    Return
    ------
    matplotlib.figure.Figure
        Figure
    matplotlib.axes.Axes
        Axes (with relevant projection)
    """
    if fig is None or ax is None:
        fig, ax = plot_fov_background(
            map_type=map_type,
            cat=cat,
            time=time,
            observer=observer,
            show=False,
            **bg_kwargs,
        )
    cat.plot_fov(ax, **kwargs)
    return _show_or_save(fig, ax, show, save)
