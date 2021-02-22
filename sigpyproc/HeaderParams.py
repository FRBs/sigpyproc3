import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class BitsInfo:
    nbits: int
    dtype: str
    digi_sigma: float = 6.0

    @property
    def itemsize(self) -> bool:
        return np.dtype(self.dtype).itemsize

    @property
    def unpack(self) -> bool:
        return bool(self.nbits in {1, 2, 4})

    @property
    def bitfact(self) -> int:
        return 8 // self.nbits if self.unpack else 1

    @property
    def digi_min(self) -> int:
        return None if self.nbits == 32 else 0

    @property
    def digi_max(self) -> int:
        return None if self.nbits == 32 else (1 << self.nbits) - 1

    @property
    def digi_mean(self) -> float:
        return None if self.nbits == 32 else (1 << (self.nbits - 1)) - 0.5

    @property
    def digi_scale(self) -> float:
        return None if self.nbits == 32 else self.digi_mean / self.digi_sigma

    def properties(self):
        return {
            key: getattr(self, key)
            for key, value in vars(type(self)).items()
            if isinstance(value, property)
        }


bits_info = {
    1: BitsInfo(1, dtype="<u1", digi_sigma=0.5),
    2: BitsInfo(2, dtype="<u1", digi_sigma=1.5),
    4: BitsInfo(4, dtype="<u1"),
    8: BitsInfo(8, dtype="<u1"),
    16: BitsInfo(16, dtype="<u2"),
    32: BitsInfo(32, dtype="<f4"),
}

# dictionary to define the sizes of header elements
header_keys = {
    "HEADER_START": None,
    "HEADER_END": None,
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
sigproc_keys = [
    "signed",
    "telescope_id",
    "ibeam",
    "nbeams",
    "refdm",
    "nifs",
    "nchans",
    "foff",
    "fch1",
    "nbits",
    "tsamp",
    "tstart",
    "src_dej",
    "src_raj",
    "za_start",
    "az_start",
    "source_name",
    "rawdatafile",
    "data_type",
    "machine_id",
]

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
data_types = {1: "Filterbank file", 2: "Timeseries file"}

# convert between types from the struct module and numpy
struct_to_numpy = {"I": "uint", "d": "float", "str": "S256"}

telescope_ids = {
    "Fake": 0,
    "Arecibo": 1,
    "Ooty": 2,
    "Nancay": 3,
    "Parkes": 4,
    "Jodrell": 5,
    "GBT": 6,
    "GMRT": 7,
    "Effelsberg": 8,
    "Effelsberg LOFAR": 9,
    "Unknown": 10,
}

ids_to_telescope = dict(zip(telescope_ids.values(), telescope_ids.keys()))

machine_ids = {
    "FAKE": 0,
    "PSPM": 1,
    "Wapp": 2,
    "AOFTM": 3,
    "BCPM1": 4,
    "OOTY": 5,
    "SCAMP": 6,
    "GBT Pulsar Spigot": 7,
    "PFFTS": 8,
    "Unknown": 9,
}

ids_to_machine = dict(zip(machine_ids.values(), machine_ids.keys()))

# not required (may be of use in future)
telescope_lats_longs = {"Effelsberg": (50.52485, 6.883593)}

# useful for creating inf files
inf_to_header = {
    "Data file name without suffix": ["basename", str],
    "Telescope used": ["telescope_id", str],
    "Instrument used": ["machine_id", str],
    "Object being observed": ["source_name", str],
    "J2000 Right Ascension (hh:mm:ss.ssss)": ["src_raj", str],
    "J2000 Declination     (dd:mm:ss.ssss)": ["src_dej", str],
    "Epoch of observation (MJD)": ["tstart", float],
    "Barycentered?           (1=yes, 0=no)": ["barycentric", int],
    "Number of bins in the time series": ["nsamples", int],
    "Width of each time series bin (sec)": ["tsamp", float],
    "Dispersion measure (cm-3 pc)": ["refdm", float],
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
    zip(psrfits_to_sigpyproc.values(), psrfits_to_sigpyproc.keys())
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
