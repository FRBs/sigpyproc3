import numpy as np
import sigpyproc.HeaderParams as conf
from os.path import splitext
from struct import pack
from sigpyproc.Utils import File

class Header(dict):
    """Container object to handle observation metadata.

    .. note::

       Attributes are mirrored as items and vice versa to facilitate cleaner code.
 
    """
    def __init__(self,info):
        super(Header,self).__init__(info)
        self._mirror()
        self.updateHeader()

    def __setitem__(self,key,value):
        self.__setattr__(key,value)

    def __setattr__(self,key,value):
        super(Header, self).__setattr__(key,value)
        super(Header, self).__setitem__(key,value)
                
    def _mirror(self):
        for key,value in self.items():
            super(Header, self).__setattr__(key,value)

    def updateHeader(self):
        """Check for changes in header and recalculate all derived quantaties.
        """
        if hasattr(self,"filename"):
            self.basename, self.extension = splitext(self.filename)
        
        if hasattr(self,"foff") and hasattr(self,"nchans") and hasattr(self,"fch1"):
            self.bandwidth  = abs(self.foff)* self.nchans
            self.ftop       = self.fch1     - 0.5*self.foff
            self.fbottom    = self.ftop     + self.foff*self.nchans
            self.fcenter    = self.ftop     + 0.5*self.foff*self.nchans
            # If fch1 is the frequency of the middle of the top 
            # channel and foff negative, this is fine.
            # However, if fch1 the frequency of the middle of the bottom channel and foff 
            # positive you should run an Filterbank.Filterbank.invertFreq on the data
        self.tobs    = self.tsamp * self.nsamples
        self.src_raj = getattr(self,"src_raj",0)
        self.src_dej = getattr(self,"src_dej",0)
        self.ra      = radec_to_str(self.src_raj)
        self.dec     = radec_to_str(self.src_dej)
        self.ra_rad  = ra_to_rad(self.ra)
        self.dec_rad = dec_to_rad(self.dec)
        self.ra_deg  = (self.ra_rad*180.)/np.pi
        self.dec_deg = (self.dec_rad*180.)/np.pi

        
        if hasattr(self,"tstart"):
            self.obs_date,self.obs_time = MJD_to_Gregorian(self.tstart)
                
        if hasattr(self,"nbits"):
            self.dtype = conf.nbits_to_dtype[self.nbits]

    def mjdAfterNsamps(self,nsamps):
        """Find the Modified Julian Date after nsamps have elapsed.

        :param nsamps: number of samples elapsed since start of observation.
        :type nsamps: int
        
        :return: Modified Julian Date
        :rtype:
        """
        return self.tstart+((nsamps*self.tsamp)/86400.)

    def newHeader(self,update_dict=None):
        """Create a new instance of :class:`~sigpyproc.Header.Header` from the current instance.
        
        :param update_dict: values to overide existing header values
        :type update_dict: :func:`dict`
        
        :return: new header information
        :rtype: :class:`~sigpyproc.Header.Header`
        """
        new = self.copy()
        if update_dict is not None:
            new.update(update_dict)
        return Header(new)

    def dedispersedHeader(self,dm):
        """Get a dedispersed version of the current header.
        
        :param dm: dispersion measure we are dedispersing to
        :type dm: float
        
        :return: A dedispersed version of the header
        :rtype: :class:`~sigpyproc.Header.Header`
        """
        return self.newHeader({'refdm':dm,
                               'nchans':1,
                               'data_type':2,
                               'nbits':32})

    
    def SPPHeader(self,back_compatible=True):
        """Get Sigproc/sigpyproc format binary header.
        
        :param back_compatible: Flag for returning Sigproc compatible header (legacy code)
        :type back_compatible: bool
        
        :return: header in binary format
        :rtype: :func:`str`
        """
        
        self.updateHeader()
        hstart  = "HEADER_START"
        hend    = "HEADER_END"
        header  = "".join([pack("I",len(hstart)),hstart])
        
        for key in self.keys():
            if back_compatible and key not in conf.sigproc_keys:
                continue
            elif not back_compatible and key not in conf.header_keys:
                continue

            if conf.header_keys[key] == "str":
                header = "".join([header,_write_string(key,self[key])])
            elif conf.header_keys[key] == "I":
                header = "".join([header,_write_int(key,self[key])])
            elif conf.header_keys[key] == "d":
                header = "".join([header,_write_double(key,self[key])])
            elif conf.header_keys[key] == "b":
                header = "".join([header,_write_char(key,self[key])])
        return "".join([header,pack("I",len(hend)),hend])

    def makeInf(self,outfile=None):
        """Make a presto format .inf file.

        :param outfile: a filename to write to. 
        :type outfile: string

        :returns: if outfile is unspecified .inf data is returned as string
        :rtype: :func:`str`
        """
        self.updateHeader()
        inf = (" Data file name without suffix          =  %s\n"%(self.basename),
               " Telescope used                         =  Effelsberg\n",
               " Instrument used                        =  PFFTS\n",
               " Object being observed                  =  %s\n"%(self.source_name),
               " J2000 Right Ascension (hh:mm:ss.ssss)  =  %s\n"%(radec_to_str(self.src_raj)),
               " J2000 Declination     (dd:mm:ss.ssss)  =  %s\n"%(radec_to_str(self.src_dej)),
               " Data observed by                       =  Robotic overlords\n",
               " Epoch of observation (MJD)             =  %.09f\n"%(self.tstart),
               " Barycentered?           (1=yes, 0=no)  =  %d\n"%(getattr(self,"barycentric",0)),
               " Number of bins in the time series      =  %d\n"%(self.nsamples),
               " Width of each time series bin (sec)    =  %.17g\n"%(self.tsamp),
               " Any breaks in the data? (1=yes, 0=no)  =  0\n",
               " Type of observation (EM band)          =  Radio\n",
               " Beam diameter (arcsec)                 =  9.22\n",
               " Dispersion measure (cm-3 pc)           =  %.03f\n"%(getattr(self,"refdm",0.0)),
               " Number of channels                     =  %d\n"%(getattr(self,"nchans",1)),
               " Data analyzed by                       =  sigpyproc\n",)
        
        if hasattr(self,"foff") and hasattr(self,"nchans") and hasattr(self,"fch1"):
            inf += (" Central freq of low channel (Mhz)      =  %.05f\n"%(self.fbottom+0.5*abs(self.foff)),
                    " Total bandwidth (Mhz)                  =  %.05f\n"%(self.bandwidth),
                    " Channel bandwidth (Mhz)                =  %.09f\n"%(abs(self.foff)))
        else:
            inf += (" Central freq of low channel (Mhz)      =  %.05f\n"%(0.0),
                    " Total bandwidth (Mhz)                  =  %.05f\n"%(0.0),
                    " Channel bandwidth (Mhz)                =  %.09f\n"%(0.0))

        if outfile == None:
            return "".join(inf)
        else:
            f = open(outfile,"w+")
            f.write("".join(inf))
            f.close()
            return None

    def getDMdelays(self,dm,in_samples=True):
        """For a given dispersion measure get the dispersive ISM delay for each frequency channel.

        :param dm: dispersion measure to calculate delays for 
        :type dm: float
        
        :param in_samples: flag to return delays as numbers of samples (def=True)
        :type in_samples: bool

        :returns: delays for each channel (highest frequency first)
        :rtype: :class:`numpy.ndarray`
        """
        self.updateHeader()
        chanFreqs  = (np.arange(self.nchans,dtype="float128")*self.foff)+self.fch1
        delays = dm * 4.148808e3 *((chanFreqs**-2)-(self.fch1**-2))
        if in_samples:
            return (delays/self.tsamp).round().astype("int32")
        else:
            return delays

    def prepOutfile(self,filename,updates=None,nbits=None,back_compatible=True):
        """Prepare a file to have sigproc format data written to it.

        :param filename: filename of new file
        :type filename: string
        
        :param updates: values to overide existing header values
        :type updates: dict
        
        :param nbits: the bitsize of data points that will written to this file (1,2,4,8,32)
        :type nbits: int
        
        :param back_compatible: flag for making file Sigproc compatible
        :type back_compatible: bool
        
        :returns: a prepared file
        :rtype: :class:`~sigpyproc.Utils.File`        
        """
        self.updateHeader()
        if nbits is None: nbits = self.nbits
        out_file = File(filename,"w+",nbits)
        new = self.newHeader(updates)
        out_file.write(new.SPPHeader(back_compatible=back_compatible))
        return out_file

def _write_string(key,value):
    return "".join([pack("I", len(key)),
                    key,pack('I',len(value)),
                    value])

def _write_int(key,value):
    return "".join([pack('I',len(key)),
                    key,pack('I',value)])

def _write_double(key,value):
    return "".join([pack('I',len(key)),
                    key,pack('d',value)])

def _write_char(key,value):
    return "".join([pack('I',len(key)),
                    key,pack('b',value)])

def radec_to_str(val):
    """Convert Sigproc format RADEC float to a string.

    :param val: Sigproc style RADEC float (eg. 124532.123)
    :type val: float
    
    :returns: 'xx:yy:zz.zzz' format string
    :rtype: :func:`str`
    """
    if val < 0:
        sign = -1
    else:
        sign = 1
    fractional,integral = np.modf(abs(val))
    xx = (integral-(integral%10000))/10000
    yy = ((integral-(integral%100))/100)-xx*100
    zz = integral - 100*yy - 10000*xx + fractional
    zz = "%07.4f"%(zz)
    return "%02d:%02d:%s"%(sign*xx,yy,zz)

def MJD_to_Gregorian(mjd):
    """Convert Modified Julian Date to the Gregorian calender.

    :param mjd: Modified Julian Date
    :type mjd float:

    :returns: date and time
    :rtype: :func:`tuple` of :func:`str`
    """
    tt = np.fmod(mjd,1)
    hh = tt*24.
    mm = np.fmod(hh,1)*60.
    ss = np.fmod(mm,1)*60.
    ss = "%08.5f"%(ss)
    j = mjd+2400000.5
    j = int(j)
    j = j - 1721119
    y = (4 * j - 1) / 146097
    j = 4 * j - 1 - 146097 * y
    d = j / 4
    j = (4 * d + 3) / 1461
    d = 4 * d + 3 - 1461 * j
    d = (d + 4) / 4
    m = (5 * d - 3) / 153
    d = 5 * d - 3 - 153 * m
    d = (d + 5) / 5
    y = 100 * y + j
    if m < 10:
        m = m + 3
    else:
        m = m - 9
        y = y + 1
    return("%02d/%02d/%02d"%(d,m,y),"%02d:%02d:%s"%(hh,mm,ss))

def rad_to_dms(rad):
    """Convert radians to (degrees, arcminutes, arcseconds)."""
    if (rad < 0.0): sign = -1
    else: sign = 1
    arc = (180/np.pi) * np.fmod(np.fabs(rad),np.pi)
    d = int(arc)
    arc = (arc - d) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    if sign==-1 and d==0:
        return (sign * d, sign * m, sign * s)
    else:
        return (sign * d, m, s)

def dms_to_rad(deg, min_, sec):
    """Convert (degrees, arcminutes, arcseconds) to radians."""
    if (deg < 0.0):
        sign = -1
    elif (deg==0.0 and (min_ < 0.0 or sec < 0.0)):
        sign = -1
    else:
        sign = 1
    return sign * (np.pi/180/60./60.) * \
        (60.0 * (60.0 * np.fabs(deg) +
                 np.fabs(min_)) + np.fabs(sec))

def dms_to_deg(deg, min_, sec):
    """Convert (degrees, arcminutes, arcseconds) to degrees."""
    return (180./np.pi) * dms_to_rad(deg, min_, sec)

def rad_to_hms(rad):
    """Convert radians to (hours, minutes, seconds)."""
    rad = np.fmod(rad, 2*np.pi)
    if (rad < 0.0): rad = rad + 2*np.pi
    arc = (12/np.pi) * rad
    h = int(arc)
    arc = (arc - h) * 60.0
    m = int(arc)
    s = (arc - m) * 60.0
    return (h, m, s)

def hms_to_rad(hour, min_, sec):
    """Convert (hours, minutes, seconds) to radians."""
    if (hour < 0.0): sign = -1
    else: sign = 1
    return sign * np.pi/12/60./60. * \
        (60.0 * (60.0 * np.fabs(hour) +
                 np.fabs(min_)) + np.fabs(sec))

def hms_to_hrs(hour, min_, sec):
    """Convert (hours, minutes, seconds) to hours."""
    return (12./np.pi) * hms_to_rad(hour, min_, sec)

def ra_to_rad(ra_string):
    """Convert right ascension string to radians."""
    h, m, s = ra_string.split(":")
    return hms_to_rad(int(h), int(m), float(s))

def dec_to_rad(dec_string):
    """Convert declination string to radians."""
    d, m, s = dec_string.split(":")
    if "-" in d and int(d)==0:
        m, s = '-'+m, '-'+s
    return dms_to_rad(int(d), int(m), float(s))

