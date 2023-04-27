"""
Short module to determine which hdu index has data in it.
"""

from astropy.io import fits

def load_hdul(file_name: str) -> fits.HDUList:
    """ Reads in the hdul for processing."""
    hdul = fits.open(load_hdul)
    return hdul

def find_data_extension(hdul: fits.HDUList) -> int:
    """ 
    Loops through the hdul and finds where the data is stored.
    Note that we are assuming that there is only one extension with data
    that means that any MES configuration will not work.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is not None:
            return i
