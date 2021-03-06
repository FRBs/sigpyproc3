import os
import struct
import numpy as np
from sigpyproc.Utils import File
from sigpyproc import HeaderParams as conf


class Header(dict):
    """Container object to handle observation metadata.

    Parameters
    ----------
    info : dict
        a dict of metadata

    Returns
    -------
    dict
        header container

    Notes
    -----
    Attributes are mirrored as items and vice versa to facilitate cleaner code.
    """

    def __init__(self, info):
        super().__init__(info)
        self._mirror()
        self.updateHeader()

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        super().__setitem__(key, value)

    def _mirror(self):
        for key, value in self.items():
            super().__setattr__(key, value)

    def updateHeader(self):
        """Check for changes in header and recalculate all derived quantaties.
        """
        if hasattr(self, "filename"):
            basename, extension = os.path.splitext(self.filename)
            self.basename = basename
            self.extension = extension

        if hasattr(self, "foff") and hasattr(self, "nchans") and hasattr(self, "fch1"):
            self.bandwidth = abs(self.foff) * self.nchans
            self.ftop = self.fch1 - 0.5 * self.foff
            self.fbottom = self.ftop + self.foff * self.nchans
            self.fcenter = self.ftop + 0.5 * self.foff * self.nchans
            # If fch1 is the frequency of the middle of the top channel and
            # foff negative, this is fine. However, if fch1 the frequency of the
            # middle of the bottom channel and foff positive you should
            # run an Filterbank.Filterbank.invertFreq on the data
        self.tobs = self.tsamp * self.nsamples
        self.src_raj = getattr(self, "src_raj", 0)
        self.src_dej = getattr(self, "src_dej", 0)
        self.ra = radec_to_str(self.src_raj)
        self.dec = radec_to_str(self.src_dej)
        self.ra_rad = ra_to_rad(self.ra)
        self.dec_rad = dec_to_rad(self.dec)
        self.ra_deg = (self.ra_rad * 180.0) / np.pi
        self.dec_deg = (self.dec_rad * 180.0) / np.pi

        if hasattr(self, "tstart"):
            obs_date, obs_time = MJD_to_Gregorian(self.tstart)
            self.obs_date = obs_date
            self.obs_time = obs_time

        if hasattr(self, "nbits"):
            self.dtype = conf.nbits_to_dtype[self.nbits]

    def mjdAfterNsamps(self, nsamps):
        """Find the Modified Julian Date after nsamps have elapsed.

        Parameters
        ----------
        nsamps : int
            number of samples elapsed since start of observation.

        Returns
        -------
        float
            Modified Julian Date
        """
        return self.tstart + ((nsamps * self.tsamp) / 86400.0)

    def newHeader(self, update_dict=None):
        """Create a new instance of :class:`~sigpyproc.Header.Header` from
        the current instance.

        Parameters
        ----------
        update_dict : dict, optional
            values to overide existing header values, by default None

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            new header information
        """
        new = self.copy()
        if update_dict is not None:
            new.update(update_dict)
        return Header(new)

    def dedispersedHeader(self, dm):
        """Get a dedispersed version of the current header.

        Parameters
        ----------
        dm : float
            dispersion measure we are dedispersing to

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            A dedispersed version of the header
        """
        return self.newHeader({'refdm': dm, 'nchans': 1, 'data_type': 2, 'nbits': 32})

    def SPPHeader(self, back_compatible=True):
        """Get Sigproc/sigpyproc format binary header.

        Parameters
        ----------
        back_compatible : bool, optional
            Flag for returning Sigproc compatible header (legacy code), by default True

        Returns
        -------
        str
            header in binary format
        """
        self.updateHeader()
        hstart = b"HEADER_START"
        hend   = b"HEADER_END"
        header = b"".join([struct.pack("I", len(hstart)), hstart])

        for key in list(self.keys()):
            if back_compatible and key not in conf.sigproc_keys:
                continue
            elif not back_compatible and key not in conf.header_keys:
                continue

            if conf.header_keys[key] == "str":
                header = b"".join([header, _write_string(key, self[key])])
            elif conf.header_keys[key] == "I":
                header = b"".join([header, _write_int(key, self[key])])
            elif conf.header_keys[key] == "d":
                header = b"".join([header, _write_double(key, self[key])])
            elif conf.header_keys[key] == "b":
                header = b"".join([header, _write_char(key, self[key])])
        return b"".join([header, struct.pack("I", len(hend)), hend])

    def makeInf(self, outfile=None):
        """Make a presto format ``.inf`` file.

        Parameters
        ----------
        outfile : str, optional
            a filename to write to, by default None

        Returns
        -------
        str
            if outfile is unspecified ``.inf`` data is returned as string
        """
        self.updateHeader()
        inf = (
            f" Data file name without suffix          =  {self.basename}\n"
            f" Telescope used                         =  Effelsberg\n"
            f" Instrument used                        =  PFFTS\n"
            f" Object being observed                  =  {self.source_name}\n"
            f" J2000 Right Ascension (hh:mm:ss.ssss)  =  {radec_to_str(self.src_raj)}\n"
            f" J2000 Declination     (dd:mm:ss.ssss)  =  {radec_to_str(self.src_dej)}\n"
            f" Data observed by                       =  Robotic overlords\n"
            f" Epoch of observation (MJD)             =  {self.tstart:.09f}\n"
            f" Barycentered?           (1=yes, 0=no)  =  {getattr(self,'barycentric',0):d}\n"
            f" Number of bins in the time series      =  {self.nsamples:d}\n"
            f" Width of each time series bin (sec)    =  {self.tsamp:.17g}\n"
            f" Any breaks in the data? (1=yes, 0=no)  =  0\n"
            f" Type of observation (EM band)          =  Radio\n"
            f" Beam diameter (arcsec)                 =  9.22\n"
            f" Dispersion measure (cm-3 pc)           =  {getattr(self, 'refdm', 0.0):.03f}\n"
            f" Number of channels                     =  {getattr(self, 'nchans', 1):d}\n"
            f" Data analyzed by                       =  sigpyproc\n"
        )

        if hasattr(self, "foff") and hasattr(self, "nchans") and hasattr(self, "fch1"):
            inf += (
                f" Central freq of low channel (Mhz)      =  {self.fbottom+0.5*abs(self.foff):.05f}\n"
                f" Total bandwidth (Mhz)                  =  {self.bandwidth:.05f}\n"
                f" Channel bandwidth (Mhz)                =  {abs(self.foff):.09f}\n"
            )
        else:
            inf += (
                f" Central freq of low channel (Mhz)      =  {0.0:.05f}\n"
                f" Total bandwidth (Mhz)                  =  {0.0:.05f}\n"
                f" Channel bandwidth (Mhz)                =  {0.0:.09f}\n"
            )

        if outfile is None:
            return inf
        with open(outfile, "w+") as f:
            f.write(inf)
        return None

    def getDMdelays(self, dm, in_samples=True):
        """For a given dispersion measure get the dispersive ISM delay
        for each frequency channel.

        Parameters
        ----------
        dm : float
            dispersion measure to calculate delays for
        in_samples : bool, optional
            flag to return delays as numbers of samples, by default True

        Returns
        -------
        :py:obj:`numpy.ndarray`
            delays for each channel (highest frequency first)
        """
        self.updateHeader()
        chanFreqs = (np.arange(self.nchans, dtype="float128") * self.foff) + self.fch1
        delays = dm * 4.148808e3 * ((chanFreqs ** -2) - (self.fch1 ** -2))
        if in_samples:
            return (delays / self.tsamp).round().astype("int32")
        return delays

    def prepOutfile(self, filename, updates=None, nbits=None, back_compatible=True):
        """Prepare a file to have sigproc format data written to it.

        Parameters
        ----------
        filename : str
            name of new file
        updates : dict, optional
            values to overide existing header values, by default None
        nbits : int, optional
            the bitsize of data points that will written to this file (1,2,4,8,32),
            by default None
        back_compatible : bool, optional
            flag for making file Sigproc compatible, by default True

        Returns
        -------
        :class:`~sigpyproc.Utils.File`
            a prepared file
        """
        self.updateHeader()
        if nbits is None:
            nbits = self.nbits
        out_file = File(filename, "w+", nbits)
        new = self.newHeader(updates)
        new["nbits"] = nbits
        out_file.write(new.SPPHeader(back_compatible=back_compatible))
        return out_file

    @classmethod
    def parseInfHeader(cls, filename):
        """Parse the metadata from a presto ``.inf`` file.

        Parameters
        ----------
        filename : str
            the name of the ``.inf`` file containing the header

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            observational metadata
        """
        header = {}
        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            key = line.split("=")[0].strip()
            val = line.split("=")[-1].strip()
            if key not in list(conf.inf_to_header.keys()):
                continue
            else:
                key, keytype = conf.inf_to_header[key]
                header[key] = keytype(val)

        header["src_raj"]      = float("".join(header["src_raj"].split(":")))
        header["src_dej"]      = float("".join(header["src_dej"].split(":")))
        header["telescope_id"] = conf.telescope_ids.get(header["telescope_id"], 10)
        header["machine_id"]   = conf.machine_ids.get(header["machine_id"], 9)
        header["data_type"]    = 2
        header["nchans"]       = 1
        header["nbits"]        = 32
        header["hdrlen"]       = 0
        return cls(header)

    @classmethod
    def parseSigprocHeader(cls, filename):
        """Parse the metadata from a Sigproc-style file header.

        Parameters
        ----------
        filename : str
            sigproc filterbank file containing the header

        Returns
        -------
        :class:`~sigpyproc.Header.Header`
            observational metadata

        Raises
        ------
        IOError
            If file header is not in sigproc format
        """
        f = open(filename, "rb")
        header = {}
        try:
            keylen = struct.unpack("I", f.read(4))[0]
        except struct.error:
            raise IOError("File Header is not in sigproc format... Is file empty?")
        key = f.read(keylen)
        if key != b"HEADER_START":
            raise IOError("File Header is not in sigproc format")
        while True:
            keylen = struct.unpack("I", f.read(4))[0]
            key = f.read(keylen)

            # convert bytestring to unicode (Python 3)
            try:
                key = key.decode()
            except UnicodeDecodeError as e:
                print(f"Could not convert to unicode: {str(e)}")

            if key not in list(conf.header_keys.keys()):
                print(f"'{key}' not recognised header key")
                return None

            if conf.header_keys[key] == "str":
                header[key] = _read_string(f)
            elif conf.header_keys[key] == "I":
                header[key] = _read_int(f)
            elif conf.header_keys[key] == "b":
                header[key] = _read_char(f)
            elif conf.header_keys[key] == "d":
                header[key] = _read_double(f)
            if key == "HEADER_END":
                break

        header["hdrlen"]   = f.tell()
        f.seek(0, 2)
        header["filelen"]  = f.tell()
        header["nbytes"]   = header["filelen"] - header["hdrlen"]
        header["nsamples"] = 8 * header["nbytes"] // header["nbits"] // header["nchans"]
        f.seek(0)
        header["filename"] = filename
        header["basename"] = os.path.splitext(filename)[0]
        f.close()
        return cls(header)


def _read_string(f):
    strlen = struct.unpack("I", f.read(4))[0]
    return f.read(strlen).decode()


def _read_int(f):
    return struct.unpack("I", f.read(4))[0]


def _read_double(f):
    return struct.unpack("d", f.read(8))[0]


def _read_char(f):
    return struct.unpack("b", f.read(1))[0]


def _write_string(key, value):
    key = key.encode()
    value = value.encode()
    return b"".join([struct.pack("I", len(key)), key,
                    struct.pack('I', len(value)), value])


def _write_int(key, value):
    key = key.encode()
    return b"".join([struct.pack('I', len(key)), key, struct.pack('I', value)])


def _write_double(key, value):
    key = key.encode()
    return b"".join([struct.pack('I', len(key)), key, struct.pack('d', value)])


def _write_char(key, value):
    key = key.encode()
    return b"".join([struct.pack('I', len(key)), key, struct.pack('b', value)])


def radec_to_str(val):
    """Convert Sigproc format RADEC float to a string.

    :param val: Sigproc style RADEC float (eg. 124532.123)
    :type val: float

    :returns: 'xx:yy:zz.zzz' format string
    :rtype: :func:`str`
    """
    sign = -1 if val < 0 else 1
    val  = np.fabs(val)
    xx   = int(val // 10000)
    yy   = int(val // 100) - xx * 100
    zz   = val - 100 * yy - 10000 * xx
    return f"{sign*xx:02d}:{yy:02d}:{zz:07.4f}"


def MJD_to_Gregorian(mjd):
    """Convert Modified Julian Date to the Gregorian calender.

    :param mjd: Modified Julian Date
    :type mjd float:

    :returns: date and time
    :rtype: :func:`tuple` of :func:`str`
    """
    hh = np.fmod(mjd, 1) * 24.0
    mm = np.fmod(hh, 1) * 60.0
    ss = np.fmod(mm, 1) * 60.0
    j = mjd + 2400000.5
    j = int(j)
    j = j - 1721119
    y = (4 * j - 1) // 146097
    j = 4 * j - 1 - 146097 * y
    d = j // 4
    j = (4 * d + 3) // 1461
    d = 4 * d + 3 - 1461 * j
    d = (d + 4) // 4
    m = (5 * d - 3) // 153
    d = 5 * d - 3 - 153 * m
    d = (d + 5) // 5
    y = 100 * y + j
    if m < 10:
        m = m + 3
    else:
        m = m - 9
        y = y + 1
    return (f"{d:02d}/{m:02d}/{y:02d}", f"{int(hh):02d}:{int(mm):02d}:{ss:08.5f}")


def rad_to_dms(rad):
    """Convert radians to (degrees, arcminutes, arcseconds)."""
    sign = -1 if rad < 0 else 1
    arc = (180 / np.pi) * np.fmod(np.fabs(rad), np.pi)
    d = int(arc)
    arc = (arc - d) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    if sign == -1 and d == 0:
        return (sign * d, sign * m, sign * s)
    else:
        return (sign * d, m, s)


def dms_to_rad(deg, min_, sec):
    """Convert (degrees, arcminutes, arcseconds) to radians."""
    if deg < 0.0:
        sign = -1
    elif deg == 0.0 and (min_ < 0.0 or sec < 0.0):
        sign = -1
    else:
        sign = 1
    return (
        sign
        * (np.pi / 180 / 60. / 60.)
        * (60.0 * (60.0 * np.fabs(deg) + np.fabs(min_)) + np.fabs(sec))
    )


def dms_to_deg(deg, min_, sec):
    """Convert (degrees, arcminutes, arcseconds) to degrees.
    """
    return (180. / np.pi) * dms_to_rad(deg, min_, sec)


def rad_to_hms(rad):
    """Convert radians to (hours, minutes, seconds).
    """
    rad = np.fmod(rad, 2 * np.pi)
    if rad < 0.0:
        rad = rad + 2 * np.pi
    arc = (12 / np.pi) * rad
    h = int(arc)
    arc = (arc - h) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    return (h, m, s)


def hms_to_rad(hour, min_, sec):
    """Convert (hours, minutes, seconds) to radians."""
    sign = -1 if hour < 0 else 1
    return (
        sign * np.pi / 12 / 60.0 / 60.0
        * (60.0 * (60.0 * np.fabs(hour) + np.fabs(min_)) + np.fabs(sec))
    )


def hms_to_hrs(hour, min_, sec):
    """Convert (hours, minutes, seconds) to hours."""
    return (12.0 / np.pi) * hms_to_rad(hour, min_, sec)


def ra_to_rad(ra_string):
    """Convert right ascension string to radians."""
    h, m, s = ra_string.split(":")
    return hms_to_rad(int(h), int(m), float(s))


def dec_to_rad(dec_string):
    """Convert declination string to radians."""
    d, m, s = dec_string.split(":")
    if "-" in d and int(d) == 0:
        m, s = "-" + m, "-" + s
    return dms_to_rad(int(d), int(m), float(s))
