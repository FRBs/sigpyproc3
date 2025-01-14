from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import constants, units
from bidict import bidict

if TYPE_CHECKING:
    from typing import Callable

DM_CONSTANT_LK = 4.148808e3  # L&K Handbook of Pulsar Astronomy
DM_CONSTANT_MT = 1 / 0.000241  # TEMPO2 Manchester & Taylor (1972)
DM_CONSTANT_SI = (
    (constants.e.esu**2 / (2 * np.pi * constants.m_e * constants.c)).to(
        units.s * units.MHz**2 * units.cm**3 / units.pc,
    )
).value  # Precise SI constants


def compute_dmdelays(
    freqs: np.ndarray,
    dm: float | np.ndarray,
    tsamp: float,
    ref_freq: float,
    *,
    in_samples: bool = True,
) -> np.ndarray:
    """Compute ISM disperison delays for given DM value(s).

    Parameters
    ----------
    freqs : ndarray
        Frequencies of each channel.
    dm : float | ndarray
        Dispersion measure(s) to calculate delays for.
    tsamp : float
        Sampling time of the data.
    ref_freq : float
        Reference frequency to calculate delays from.
    in_samples : bool, optional
        Flag to return delays as numbers of samples, by default True.

    Returns
    -------
    ndarray
        Dispersion delays at middle of each channel with respect to reference frequency.

        If dm is a scalar, returns a 1D array of delays. If dm is an array,
        returns a 2D array with shape ``(len(dm), len(freqs))``.
    """
    dm = np.atleast_1d(dm)[:, np.newaxis]
    delays = dm * DM_CONSTANT_LK * ((freqs**-2) - (ref_freq**-2))
    if in_samples:
        delays = (delays / tsamp).round().astype(np.int32)
    return delays.squeeze()


def compute_dmsmearing(
    freqs: np.ndarray,
    dm: float | np.ndarray,
    tsamp: float,
    *,
    in_samples: bool = True,
) -> np.ndarray:
    """Compute ISM smearing due to finite bandwidth for given DM value(s).

    Parameters
    ----------
    freqs : ndarray
        Frequencies of each channel.
    dm : float | ndarray
        Dispersion measure(s) to calculate smearing for.
    tsamp : float
        Sampling time of the data.
    in_samples : bool, optional
        Flag to return smearing as numbers of samples, by default True.

    Returns
    -------
    ndarray
        DM smearing in the frequency channels due to finite bandwidth.

        If dm is a scalar, returns a 1D array of smearing. If dm is an array,
        returns a 2D array with shape ``(len(dm), len(freqs))``.
    """
    dm = np.atleast_1d(dm)[:, np.newaxis].astype(np.float32)
    foff = np.abs(freqs[1] - freqs[0])
    smearing = 2 * DM_CONSTANT_LK * foff * dm / freqs**3
    if in_samples:
        smearing = (smearing / tsamp).round().astype(np.int32)
    return smearing.squeeze()


# dictionary to define the sizes of header elements
header_keys = {
    "filename": "str",
    "telescope_id": "I",
    "telescope": "str",
    "machine_id": "I",
    "data_type": "I",
    "rawdatafile": "str",
    "source_name": "str",
    "barycentric": "I",
    "pulsarcentric": "I",
    "az_start": "d",
    "za_start": "d",
    "src_raj": "d",
    "src_dej": "d",
    "tstart": "d",
    "tsamp": "d",
    "nbits": "I",
    "nsamples": "I",
    "fch1": "d",
    "foff": "d",
    "fchannel": "d",
    "nchans": "I",
    "nifs": "I",
    "refdm": "d",
    "flux": "d",
    "period": "d",
    "nbeams": "I",
    "ibeam": "I",
    "hdrlen": "I",
    "pb": "d",
    "ecc": "d",
    "asini": "d",
    "orig_hdrlen": "I",
    "new_hdrlen": "I",
    "sampsize": "I",
    "bandwidth": "d",
    "fbottom": "d",
    "ftop": "d",
    "obs_date": "str",
    "obs_time": "str",
    "signed": "b",
    "accel": "d",
}

# header keys recognised by the sigproc package

# header units for fancy printing
header_units = {
    "az_start": "(Degrees)",
    "za_start": "(Degrees)",
    "src_raj": "(HH:MM:SS.ss)",
    "src_dej": "(DD:MM:SS.ss)",
    "tstart": "(MJD)",
    "tsamp": "(s)",
    "fch1": "(MHz)",
    "foff": "(MHz)",
    "fchannel": "(MHz)",
    "refdm": "(pccm^-3)",
    "period": "(s)",
    "pb": "(hrs)",
    "flux": "(mJy)",
    "hdrlen": "(Bytes)",
    "orig_hdrlen": "(Bytes)",
    "new_hdrlen": "(Bytes)",
    "sampsize": "(Bytes)",
    "bandwidth": "(MHz)",
    "fbottom": "(MHz)",
    "ftop": "(MHz)",
    "ra_rad": "(Radians)",
    "dec_rad": "(Radians)",
    "ra_deg": "(Degrees)",
    "dec_deg": "(Degrees)",
    "Glon": "(Degrees)",
    "Glat": "(Degrees)",
    "obs_date": "(dd/mm/yy)",
    "obs_time": "(hh:mm:ss.sssss)",
}

# data type flag for sigproc files
data_types = bidict(
    {
        0: "raw data",
        1: "filterbank",
        2: "time series",
        3: "pulse profiles",
        4: "amplitude spectrum",
        5: "complex spectrum",
        6: "dedispersed subbands",
        10: "PSRFITS",
    },
)

# convert between types from the struct module and numpy
struct_to_numpy = {"I": "uint", "d": "float", "str": "S256"}


# not required (may be of use in future)
telescope_lats_longs = {"Effelsberg": (50.52485, 6.883593)}

# useful for creating inf files
presto_inf: dict[str, tuple[str, Callable, str]] = {
    "Data file name without suffix": ("basename", str, "s"),
    "Telescope used": ("telescope", str, "s"),
    "Instrument used": ("backend", str, "s"),
    "Object being observed": ("source", str, "s"),
    "J2000 Right Ascension (hh:mm:ss.ssss)": ("ra", str, "s"),
    "J2000 Declination     (dd:mm:ss.ssss)": ("dec", str, "s"),
    "Data observed by": ("observer", str, "s"),
    "Epoch of observation (MJD)": ("tstart", float, "05.15f"),
    "Barycentered?           (1=yes, 0=no)": ("barycentric", int, "d"),
    "Number of bins in the time series": ("nsamples", int, "-11.0f"),
    "Width of each time series bin (sec)": ("tsamp", float, ".15g"),
    "Dispersion measure (cm-3 pc)": ("dm", float, ".12g"),
    "Central freq of low channel (MHz)": ("freq_low", float, ".12g"),
    "Total bandwidth (Mhz)": ("bandwidth", float, ".12g"),
    "Number of channels": ("nchans", int, "d"),
    "Channel bandwidth (MHz)": ("foff", float, ".12g"),
    "Data analyzed by": ("analyser", str, "s"),
}

# this could be expanded to begin adding support for PSRFITS
psrfits_to_sigpyproc = {
    "IBEAM": "ibeam",
    "NBITS": "nbits",
    "OBSNCHAN": "nchans",
    "SRC_NAME": "source_name",
    "RA": "src_raj",
    "DEC": "src_dej",
    "CHAN_BW": "foff",
    "TBIN": "tsamp",
}

sigpyproc_to_psrfits = dict(
    zip(psrfits_to_sigpyproc.values(), psrfits_to_sigpyproc.keys()),
)

sigproc_to_tempo = {0: "g", 1: "3", 3: "f", 4: "7", 6: "1", 8: "g", 5: "8"}

tempo_params = [
    "RA",
    "DEC",
    "PMRA",
    "PMDEC",
    "PMRV",
    "BETA",
    "LAMBDA",
    "PMBETA",
    "PMLAMBDA",
    "PX",
    "PEPOCH",
    "POSEPOCH",
    "F0",
    "F",
    "F1",
    "F2",
    "Fn",
    "P0",
    "P",
    "P1",
    "DM",
    "DMn",
    "A1_n",
    "E_n",
    "T0_n",
    "TASC",
    "PB_n",
    "OM_n",
    "FB",
    "FB_n",
    "FBJ_n",
    "TFBJ_n",
    "EPS1",
    "EPS2",
    "EPS1DOT",
    "EPS2DOT",
    "OMDOT",
    "OM2DOT",
    "XOMDOT",
    "PBDOT",
    "XPBDOT",
    "GAMMA",
    "PPNGAMMA",
    "SINI",
    "MTOT",
    "M2",
    "DR",
    "DTHETA",
    "XDOT",
    "XDOT_n",
    "X2DOT",
    "EDOT",
    "AFAC",
    "A0",
    "B0",
    "BP",
    "BPP",
    "GLEP_n",
    "GLPH_n",
    "GLF0_n",
    "GLF1_n",
    "GLF0D_n",
    "GLDT_n",
    "JUMP_n",
]
