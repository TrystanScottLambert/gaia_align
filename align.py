"""
Module to overplot gaia stars on an image and then use them to
determine a WCS.
"""

from typing import Tuple
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs import utils
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
import astropy.units as u
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

from draggable_scatter import DraggableScatter
from mes_check import find_data_extension, get_pixscale_from_wcs

BOX_PADDING = 20

def search_gaia_archives(ra: float, dec: float, height:float, width:float)\
    -> Tuple[np.ndarray, np.ndarray]:
    """
    Does a box search centered on ra and dec with a given height and width.
    """
    coord = SkyCoord(ra = ra*u.deg, dec=dec*u.deg)
    width = u.Quantity(width*u.deg)
    height = u.Quantity(height*u.deg)
    results = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    return np.array(list(results['ra'])), np.array(list(results['dec']))

def sum_columns_and_rows(array_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Sums up the rows and the columns so that we can fit gaussians to them."""
    column_sum = np.sum(array_2d,axis=0) # axis 0 gives columnsaccurate_y_position
    row_sum = np.sum(array_2d,axis=1)
    return column_sum, row_sum

def gauss(x_array: np.ndarray, constant:float, amplitude:float, mean:float, sigma:float)\
      -> np.ndarray:
    """Definiton of a gaussian."""
    return constant + amplitude * np.exp(-(x_array - mean) ** 2 / (2 * sigma ** 2))

def gauss_fit(x_array: np.ndarray, y_array:np.ndarray) -> np.ndarray:
    """Fit a 1-D gaussian to x and y data."""
    mean = np.sum(x_array * y_array) / np.sum(y_array)
    sigma = np.sqrt(np.sum(y_array * (x_array - mean) ** 2) / np.sum(y_array))
    popt, _ = curve_fit(gauss, x_array, y_array, p0=[np.min(y_array), np.max(y_array), mean, sigma])
    return popt

def get_star_position(star: np.ndarray) -> tuple[float, float]:
    """Determines the two 1-d gaussian fit positoin of the cut out star."""
    column_sum, row_sum = sum_columns_and_rows(star)
    x_values = np.arange(len(row_sum))
    row_popt = gauss_fit(x_values,row_sum)
    column_popt = gauss_fit(x_values,column_sum)

    return column_popt[2], row_popt[2]

def create_postage_stamp(x_position: float, y_position: float, image_data: np.ndarray) -> np.ndarray:
    """
    Cuts out a postage stamp of the data around the x_pos and y_pos. 
    """
    x_position_rounded, y_position_rounded = int(round(x_position)), int(round(y_position))
    image_postage_stamp = image_data[
         y_position_rounded-BOX_PADDING:y_position_rounded+BOX_PADDING,
         x_position_rounded-BOX_PADDING:x_position_rounded+BOX_PADDING]
    return image_postage_stamp

def cut_star(x_position: float, y_position: float, image_data: np.ndarray)\
      -> tuple[np.ndarray, float, float]:
    """Cuts out star and determines the center of said star."""
    image_postage_stamp = create_postage_stamp(x_position, y_position, image_data)

    if 0 in image_postage_stamp.shape:
        return None

    try:
        local_x_position, local_y_position = get_star_position(image_postage_stamp)
        accurate_x_position = int(x_position) - BOX_PADDING + local_x_position
        accurate_y_position = int(y_position) - BOX_PADDING + local_y_position
        return image_postage_stamp, accurate_x_position, accurate_y_position
    except (RuntimeError, ValueError, TypeError):
        return None


def get_usable_gaia(gaia_x_array: np.ndarray, gaia_y_array: np.ndarray, image: np.ndarray):
    """
    Searches the positions of the x and y coordinates for
    a bright star that can be fit.
    """

    accurate_x = []
    accurate_y = []
    indicies = []
    for i, _ in enumerate(gaia_x_array):
        star_cutout = cut_star(gaia_x_array[i], gaia_y_array[i], image)
        if star_cutout is not None:
            _, x_point, y_point = star_cutout
            accurate_x.append(x_point)
            accurate_y.append(y_point)
            indicies.append(i)

    return np.array(accurate_x), np.array(accurate_y), np.array(indicies)


class Image:
    """Main class for images."""

    def __init__(self, chip_name:str):
        """Initilizing."""
        self.file_name = chip_name
        self.hdul = fits.open(chip_name)
        self.data_extension = find_data_extension(self.hdul)
        self.data = self.hdul[self.data_extension].data
        self.header = self.hdul[self.data_extension].header
        self.current_wcs = WCS(self.header)
        current_gaia_info = self.query_gaia()
        self.gaia_ra = current_gaia_info[0]
        self.gaia_dec = current_gaia_info[1]
        self.current_gaia_x = current_gaia_info[2]
        self.current_gaia_y = current_gaia_info[3]
        interval = ZScaleInterval()
        self.vmin, self.vmax = interval.get_limits(self.data)

    def query_gaia(self):
        """Gets the gaia positions in the chip frame."""
        center_y_pix = self.data.shape[0]/2
        center_x_pix = self.data.shape[1]/2

        # Assuming square pixels.
        search_width = self.data.shape[1] * get_pixscale_from_wcs(self.header)
        search_height = self.data.shape[0] * get_pixscale_from_wcs(self.header)

        center_ra, center_dec = self.current_wcs.pixel_to_world_values(center_x_pix, center_y_pix)
        gaia_ra, gaia_dec = search_gaia_archives(
            center_ra, center_dec, width = search_width, height=search_height)
        gaia_x, gaia_y = self.current_wcs.world_to_pixel_values(gaia_ra, gaia_dec)

        return gaia_ra, gaia_dec, gaia_x, gaia_y

    def _manually_determine_gaia_offsets(self):
        """
        Starts the draggable plot for alignment and uses user input to determine
        the pixel offsets of the GAIA stars.
        """

        fig = plt.figure()
        ax_align = fig.add_subplot(projection=self.current_wcs)
        ax_align.set_title(self.file_name)
        ax_align.imshow(self.data, vmin=self.vmin, vmax=self.vmax, cmap='gray')
        scatter = ax_align.scatter(
            self.current_gaia_x, self.current_gaia_y, facecolor='None',
            edgecolor='r', marker='s', s=80, picker=True)
        DraggableScatter(scatter)
        plt.show()
        updated_gaia_x, updated_gaia_y = scatter.get_offsets().T
        offset_x = updated_gaia_x - self.current_gaia_x
        offset_y = updated_gaia_y - self.current_gaia_y
        return offset_x, offset_y

    def _overplot_scatter(self, x_scatter: np.ndarray, y_scatter: np.ndarray):
        """
        Plots scatter points over the chip data.
        """
        fig = plt.figure()
        ax_scatter = fig.add_subplot(projection=self.current_wcs)
        ax_scatter.set_title(self.file_name)
        ax_scatter.imshow(self.data, vmin=self.vmin, vmax=self.vmax, cmap='gray')
        ax_scatter.scatter(
            x_scatter, y_scatter, facecolor='None', edgecolor='r', marker='s', s=100, picker=True)
        plt.show()

    def _fit_gaia_coords(self, gaia_offset_x: np.ndarray, gaia_offset_y: np.ndarray) \
          -> tuple[SkyCoord, np.ndarray]:
        """
        Uses offset to align gaia points then fits the these points accurately
        returning their onsky position and their matching xy values.
        """
        updated_gaia_x = self.current_gaia_x + gaia_offset_x
        updated_gaia_y = self.current_gaia_y + gaia_offset_y
        accurate_gaia_x, accurate_gaia_y, msk = get_usable_gaia(
            updated_gaia_x, updated_gaia_y, self.data)

        diff = np.hypot(updated_gaia_x[msk]-accurate_gaia_x, updated_gaia_y[msk]-accurate_gaia_y)
        _, diff_median, diff_std = sigma_clipped_stats(diff)
        cut = np.where(diff < diff_median + 1*diff_std)[0]

        self.gaia_offset_x = np.mean(accurate_gaia_x[cut] - self.current_gaia_x[msk][cut])
        self.gaia_offset_y = np.mean(accurate_gaia_y[cut] - self.current_gaia_y[msk][cut])

        gaia_coords = [(self.gaia_ra[msk][cut][i], self.gaia_dec[msk][cut][i]) \
                        for i, _ in enumerate(self.gaia_ra[msk][cut])]
        gaia_skycoords = SkyCoord(gaia_coords, unit=(u.deg, u.deg))
        gaia_pixcoords = np.array([accurate_gaia_x[cut], accurate_gaia_y[cut]])
        return gaia_skycoords, gaia_pixcoords


    def determine_wcs_manually(self, show_alignment: bool = False) -> WCS:
        """
        Using draggabale scatter plot to determine the x,y offsets.
        """
        gaia_offset_x, gaia_offset_y = self._manually_determine_gaia_offsets()
        gaia_skycoords, gaia_pixcoords = self._fit_gaia_coords(gaia_offset_x, gaia_offset_y)

        if show_alignment:
            self._overplot_scatter(gaia_pixcoords[0], gaia_pixcoords[1])

        wcs = utils.fit_wcs_from_points(gaia_pixcoords, gaia_skycoords)

        return wcs

    def update_wcs(self, wcs_object: WCS) -> None:
        """
        Updates the chip with the given wcs object.
        """
        self.hdul[self.data_extension].header.update(wcs_object.to_header())
        self.hdul.writeto(self.file_name.split('.fits')[0] + '.wcs_aligned.fits', overwrite=True)


if __name__ == '__main__':
    INFILE = 'hz7_cosweb/HZ7_cosweb_30arcsec_f115w_nup_i2d.fits'
    image = Image(INFILE)