#include <cstring>
#include <type_traits>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <libUtils.hpp>

namespace py = pybind11;

#define HI4BITS 240
#define LO4BITS 15
#define HI2BITS 192
#define UPMED2BITS 48
#define LOMED2BITS 12
#define LO2BITS 3

/*----------------------------------------------------------------------------*/

/**
 * Function to unpack 1,2 and 4 bit data
 * data is unpacked into an empty buffer
 * Note: Only unpacks big endian bit ordering
 */
py::array_t<uint8_t> unpack(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf  = inarray.request();
    int nbytes = inbuf.size;

    auto outarray = py::array_t<uint8_t>(inbuf.size * 8 / nbits);
    py::buffer_info outbuf   = outarray.request();

    uint8_t* indata  = (uint8_t*)inbuf.ptr;
    uint8_t* outdata = (uint8_t*)outbuf.ptr;

    switch (nbits) {
        case 1:
            for (int ii = 0; ii < nbytes; ii++) {
                for (int jj = 0; jj < 8; jj++) {
                    outdata[(ii * 8) + jj] = (indata[ii] >> jj) & 1;
                }
            }
            break;
        case 2:
            for (int ii = 0; ii < nbytes; ii++) {
                outdata[(ii * 4) + 3] = indata[ii] & LO2BITS;
                outdata[(ii * 4) + 2] = (indata[ii] & LOMED2BITS) >> 2;
                outdata[(ii * 4) + 1] = (indata[ii] & UPMED2BITS) >> 4;
                outdata[(ii * 4) + 0] = (indata[ii] & HI2BITS) >> 6;
            }
            break;
        case 4:
            for (int ii = 0; ii < nbytes; ii++) {
                outdata[(ii * 2) + 1] = indata[ii] & LO4BITS;
                outdata[(ii * 2) + 0] = (indata[ii] & HI4BITS) >> 4;
            }
            break;
    }
    return outarray;
}

/**
 * Function to unpack 1,2 and 4 bit data
 * Data is unpacked into the same buffer. This is done by unpacking the bytes
 * backwards so as not to overwrite any of the data. This is old code that is
 * no longer used should the filterbank reader ever be changed from using
 * np.fromfile this may once again become useful
 * Note: Only set up for big endian bit ordering
 */
void unpackInPlace(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf  = inarray.request();
    int nbytes = inbuf.size;

    uint8_t* buffer = (uint8_t*)inbuf.ptr;

    int     pos;
    int     lastsamp = nbits * nbytes / 8;
    uint8_t temp;

    switch (nbits) {
        case 1:
            for (int ii = lastsamp - 1; ii > -1; ii--) {
                temp = buffer[ii];
                pos  = ii * 8;
                for (int jj = 0; jj < 8; jj++) {
                    buffer[pos + jj] = (temp >> jj) & 1;
                }
            }
            break;
        case 2:
            for (int ii = lastsamp - 1; ii > -1; ii--) {
                temp            = buffer[ii];
                pos             = ii * 4;
                buffer[pos + 3] = temp & LO2BITS;
                buffer[pos + 2] = (temp & LOMED2BITS) >> 2;
                buffer[pos + 1] = (temp & UPMED2BITS) >> 4;
                buffer[pos + 0] = (temp & HI2BITS) >> 6;
            }
            break;
        case 4:
            for (int ii = lastsamp - 1; ii > -1; ii--) {
                temp            = buffer[ii];
                pos             = ii * 2;
                buffer[pos + 0] = temp & LO4BITS;
                buffer[pos + 1] = (temp & HI4BITS) >> 4;
            }
            break;
    }
}

/**
 * Function to pack bit data into an empty buffer
 */
py::array_t<uint8_t> pack(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf  = inarray.request();
    int nbytes = inbuf.size;

    auto outarray = py::array_t<uint8_t>(inbuf.size * nbits / 8);
    py::buffer_info outbuf   = outarray.request();

    uint8_t* indata  = (uint8_t*)inbuf.ptr;
    uint8_t* outdata = (uint8_t*)outbuf.ptr;

    int     pos;
    int     bitfact = 8 / nbits;
    uint8_t val;

    switch (nbits) {
        case 1:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 8;
                val = (indata[pos + 7] << 7) | (indata[pos + 6] << 6) |
                      (indata[pos + 5] << 5) | (indata[pos + 4] << 4) |
                      (indata[pos + 3] << 3) | (indata[pos + 2] << 2) |
                      (indata[pos + 1] << 1) | indata[pos + 0];
                outdata[ii] = val;
            }
            break;
        case 2:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 4;
                val = (indata[pos] << 6) | (indata[pos + 1] << 4) |
                      (indata[pos + 2] << 2) | indata[pos + 3];
                outdata[ii] = val;
            }
            break;
        case 4:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 2;
                val = (indata[pos] << 4) | indata[pos + 1];

                outdata[ii] = val;
            }
            break;
    }
    return outarray;
}

/**
 * Function to pack bit data into the same buffer
 */
void packInPlace(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf  = inarray.request();
    int nbytes = inbuf.size;

    uint8_t* buffer = (uint8_t*)inbuf.ptr;

    int     pos;
    int     bitfact = 8 / nbits;
    uint8_t val;

    switch (nbits) {
        case 1:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 8;
                val = (buffer[pos + 7] << 7) | (buffer[pos + 6] << 6) |
                      (buffer[pos + 5] << 5) | (buffer[pos + 4] << 4) |
                      (buffer[pos + 3] << 3) | (buffer[pos + 2] << 2) |
                      (buffer[pos + 1] << 1) | buffer[pos + 0];
                buffer[ii] = val;
            }
            break;
        case 2:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 4;
                val = (buffer[pos] << 6) | (buffer[pos + 1] << 4) |
                      (buffer[pos + 2] << 2) | buffer[pos + 3];
                buffer[ii] = val;
            }
            break;
        case 4:
            for (int ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 2;
                val = (buffer[pos] << 4) | buffer[pos + 1];

                buffer[ii] = val;
            }
            break;
    }
}

/**
 * Convert 1-,2- or 4-bit data to 8-bit data and write to file.
 */
void to8bit(py::array_t<float> inarray, py::array_t<uint8_t> outarray,
            py::array_t<uint8_t> flag, py::array_t<float> fact,
            py::array_t<float> plus, py::array_t<float> flagMax,
            py::array_t<float> flagMin, int nsamps, int nchans) {

    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    flagbuf = flag.request(), factbuf = fact.request(),
                    plusbuf = plus.request(), flagMaxbuf = flagMax.request(),
                    flagMinbuf = flagMin.request();

    float*   indata      = (float*)inbuf.ptr;
    uint8_t* outdata     = (uint8_t*)outbuf.ptr;
    uint8_t* flag_arr    = (uint8_t*)flagbuf.ptr;
    float*   fact_arr    = (float*)factbuf.ptr;
    float*   plus_arr    = (float*)plusbuf.ptr;
    float*   flagMax_arr = (float*)flagMaxbuf.ptr;
    float*   flagMin_arr = (float*)flagMinbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outdata[(ii * nchans) + jj] =
                indata[(ii * nchans) + jj] / fact_arr[jj] - plus_arr[jj];
            if (indata[(ii * nchans) + jj] > flagMax_arr[jj])
                flag_arr[(ii * nchans) + jj] = 2;
            else if (indata[(ii * nchans) + jj] < flagMin_arr[jj])
                flag_arr[(ii * nchans) + jj] = 0;
            else
                flag_arr[(ii * nchans) + jj] = 1;
        }
    }
}


/**
 * TODO: This code is very slow compared to python slicing.
 */
template <class T>
void splitToBands(py::array_t<T> inarray, py::array_t<T> outarray, int nchans,
                  int nsamps, int nsub, int chanpersub, int chanstart) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

    int chan_end = nsub * chanpersub + chanstart;
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = chanstart; jj < chan_end; jj++) {
            outdata[(ii * nchans) +
                    (jj % chanpersub * nsub + jj / chanpersub)] =
                indata[(ii * nchans) + jj];
        }
    }
}

template <class T>
void getTim(py::array_t<T> inarray, py::array_t<float> outarray, int nchans,
            int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T*     indata  = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outdata[index + ii] += indata[(nchans * ii) + jj];
        }
    }
}

template <class T>
void getBpass(py::array_t<T> inarray, py::array_t<double> outarray, int nchans,
              int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T*      indata  = (T*)inbuf.ptr;
    double* outdata = (double*)outbuf.ptr;
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        for (int ii = 0; ii < nsamps; ii++) {
            outdata[jj] += indata[(nchans * ii) + jj];
        }
    }
}

template <class T>
void dedisperse(py::array_t<T> inarray, py::array_t<float> outarray,
                py::array_t<int32_t> delays, int maxdelay, int nchans,
                int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();
    py::buffer_info delaysbuf = delays.request();

    T*       indata     = (T*)inbuf.ptr;
    float*   outdata    = (float*)outbuf.ptr;
    int32_t* delays_arr = (int32_t*)delaysbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outdata[index + ii] +=
                indata[(ii * nchans) + (delays_arr[jj] * nchans) + jj];
        }
    }
}

template <class T>
void maskChannels(py::array_t<T> inarray, py::array_t<uint8_t> mask, int nchans,
                  int nsamps) {
    py::buffer_info inbuf = inarray.request(), maskbuf = mask.request();

    T*       indata   = (T*)inbuf.ptr;
    uint8_t* mask_arr = (uint8_t*)maskbuf.ptr;

    for (int ii = 0; ii < nchans; ii++) {
        if (mask_arr[ii] == 0) {
            for (int jj = 0; jj < nsamps; jj++) {
                indata[jj * nchans + ii] = 0.0;
            }
        }
    }
}

template <class T>
void subband(py::array_t<T> inarray, py::array_t<float> outarray,
             py::array_t<int32_t> delays, py::array_t<int32_t> c_to_s,
             int maxdelay, int nchans, int nsubs, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();
    py::buffer_info delaysbuf = delays.request();
    py::buffer_info c_to_sbuf = c_to_s.request();

    T*       indata     = (T*)inbuf.ptr;
    float*   outdata    = (float*)outbuf.ptr;
    int32_t* delays_arr = (int32_t*)delaysbuf.ptr;
    int32_t* c_to_s_arr = (int32_t*)c_to_sbuf.ptr;

    int out_size;
    out_size = nsubs * (nsamps - maxdelay) * sizeof(float);
    std::memset(outdata, 0.0, out_size);

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outdata[(ii * nsubs) + c_to_s_arr[jj]] +=
                (float)indata[(ii * nchans) + (delays_arr[jj] * nchans) + jj];
        }
    }
}

template <class T>
void getChan(py::array_t<T> inarray, py::array_t<float> outarray, int chan,
             int nchans, int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T*     indata  = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++)
        outdata[index + ii] = indata[(ii * nchans) + chan];
}

template <class T>
void splitToChans(py::array_t<T> inarray, py::array_t<float> outarray,
                  int nchans, int nsamps, int gulp) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T*     indata  = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++)
            outdata[(jj * gulp) + ii] = indata[(ii * nchans) + jj];
    }
}

template <class T>
void invertFreq(py::array_t<T> inarray, py::array_t<T> outarray, int nchans,
                int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outdata[(jj) + ii * nchans] =
                indata[(nchans - 1 - jj) + ii * nchans];
        }
    }
}

template <class T>
void foldFil(py::array_t<T> inarray, py::array_t<float> foldarray,
             py::array_t<int32_t> countarray, py::array_t<int32_t> delays,
             int maxDelay, double tsamp, double period, double accel,
             int totnsamps, int nsamps, int nchans, int nbins, int nints,
             int nsubs, int index) {
    py::buffer_info inbuf = inarray.request(), foldbuf = foldarray.request();
    py::buffer_info countbuf  = countarray.request();
    py::buffer_info delaysbuf = delays.request();

    T*       indata    = (T*)inbuf.ptr;
    float*   fold_arr  = (float*)foldbuf.ptr;
    int32_t* count_arr = (int32_t*)countbuf.ptr;
    int32_t* delay_arr = (int32_t*)delaysbuf.ptr;

    int   tobs, phasebin, subband, subint, pos1, pos2;
    float factor1, factor2, val, tj;
    float c = 299792458.0;
    factor1 = (float)totnsamps / nints;
    factor2 = (float)nchans / nsubs;
    tobs    = (int)(totnsamps * tsamp);
    for (int ii = 0; ii < (nsamps - maxDelay); ii++) {
        tj = (ii + index) * tsamp;
        phasebin =
            ((int)(nbins * tj * (1 + accel * (tj - tobs) / (2 * c)) / period +
                   0.5)) %
            nbins;
        subint = (int)((index + ii) / factor1);
        pos1   = (subint * nsubs * nbins) + phasebin;
        for (int jj = 0; jj < nchans; jj++) {
            subband = (int)(jj / factor2);
            pos2    = pos1 + (subband * nbins);
            val     = indata[(ii * nchans) + (delay_arr[jj] * nchans) + jj];
            fold_arr[pos2] += val;
            count_arr[pos2]++;
        }
    }
}

/**
 * getStats: Computing central moments in one pass through the data,
 * the algorithm is numerically stable and accurate.
 * Ref:
 * https://www.johndcook.com/blog/skewness_kurtosis/
 * https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
 */
template <class T>
void getStats(py::array_t<T> inarray, py::array_t<float> M1,
              py::array_t<float> M2, py::array_t<float> M3,
              py::array_t<float> M4, py::array_t<float> maxima,
              py::array_t<float> minima, py::array_t<int64_t> count, int nchans,
              int nsamps, int startflag) {

    py::buffer_info inbuf = inarray.request(), M1buf = M1.request(),
                    M2buf = M2.request(), M3buf = M3.request(),
                    M4buf = M4.request(), maxbuf = maxima.request(),
                    minbuf = minima.request(), countbuf = count.request();

    T*       indata    = (T*)inbuf.ptr;
    float*   M1_arr    = (float*)M1buf.ptr;
    float*   M2_arr    = (float*)M2buf.ptr;
    float*   M3_arr    = (float*)M3buf.ptr;
    float*   M4_arr    = (float*)M4buf.ptr;
    float*   max_arr   = (float*)maxbuf.ptr;
    float*   min_arr   = (float*)minbuf.ptr;
    int32_t* count_arr = (int32_t*)countbuf.ptr;

    T val;
    if (startflag == 0) {
        for (int jj = 0; jj < nchans; jj++) {
            max_arr[jj] = indata[jj];
            min_arr[jj] = indata[jj];
        }
    }
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        double delta, delta_n, delta_n2, term1;
        for (int ii = 0; ii < nsamps; ii++) {
            val = indata[(nchans * ii) + jj];
            count_arr[jj] += 1;
            long long n = count_arr[jj];

            delta    = val - M1_arr[jj];
            delta_n  = delta / n;
            delta_n2 = delta_n * delta_n;
            term1    = delta * delta_n * (n - 1);
            M1_arr[jj] += delta_n;
            M4_arr[jj] += term1 * delta_n2 * (n * n - 3 * n + 3) +
                          6 * delta_n2 * M2_arr[jj] - 4 * delta_n * M3_arr[jj];
            M3_arr[jj] += term1 * delta_n * (n - 2) - 3 * delta_n * M2_arr[jj];
            M2_arr[jj] += term1;

            if (val > max_arr[jj])
                max_arr[jj] = val;
            else if (val < min_arr[jj])
                min_arr[jj] = val;
        }
    }
}

/**
 * Digitizing code taken from SigProcDigitizer.C (dspsr)
 */
template <class T>
void removeBandpass(py::array_t<T> inarray, py::array_t<T> outarray,
                    py::array_t<float> means, py::array_t<float> stdevs,
                    int nchans, int nsamps) {

    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    meansbuf = means.request(), stdevsbuf = stdevs.request();

    T*     indata     = (T*)inbuf.ptr;
    T*     outdata    = (T*)outbuf.ptr;
    float* means_arr  = (float*)meansbuf.ptr;
    float* stdevs_arr = (float*)stdevsbuf.ptr;

    T     val;
    float DIGI_MEAN  = 127.5;
    float DIGI_SIGMA = 6;
    float DIGI_SCALE = DIGI_MEAN / DIGI_SIGMA;
    int   DIGI_MIN   = 0;
    int   DIGI_MAX   = 255;

#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        for (int ii = 0; ii < nsamps; ii++) {
            val = indata[(nchans * ii) + jj];

            double scale;
            if (stdevs_arr[jj] == 0.0)
                scale = 1.0;
            else
                scale = 1.0 / stdevs_arr[jj];

            // Normalize the data per channel to N(0,1)
            double normval = (val - means_arr[jj]) * scale;

            if (std::is_same<T, unsigned char*>::value) {
                // Shift the data for digitization
                normval = ((normval * DIGI_SCALE) + DIGI_MEAN + 0.5);
                // clip the normval at the limits
                if (normval < DIGI_MIN)
                    normval = DIGI_MIN;

                if (normval > DIGI_MAX)
                    normval = DIGI_MAX;
            }
            outdata[(nchans * ii) + jj] = (T)normval;
        }
    }
}

template <class T>
void downsample(py::array_t<T> inarray, py::array_t<T> outarray, int tfactor,
                int ffactor, int nchans, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

    int newnsamps = nsamps / tfactor;
    int newnchans = nchans / ffactor;
    int totfactor = ffactor * tfactor;
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < newnsamps; ii++) {
        for (int jj = 0; jj < newnchans; jj++) {
            int   pos  = nchans * ii * tfactor + jj * ffactor;
            float temp = 0;
            for (int kk = 0; kk < tfactor; kk++) {
                for (int ll = 0; ll < ffactor; ll++) {
                    temp += indata[kk * nchans + ll + pos];
                }
            }
            float result = temp / totfactor;
            if (std::is_same<T, uint8_t*>::value) {
                result = result + 0.5;
            }
            outdata[ii * newnchans + jj] = (T)(result);
        }
    }
}

/**
 * Remove the channel-weighted zero-DM (Eatough, Keane & Lyne 2009)
 * code based on zerodm.c (presto)
 */
template <class T>
void removeZeroDM(py::array_t<T> inarray, py::array_t<T> outarray,
                  py::array_t<float> bpass, py::array_t<float> chanwts,
                  int nchans, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    bpassbuf = bpass.request(), chanwtsbuf = chanwts.request();

    T*     indata      = (T*)inbuf.ptr;
    T*     outdata     = (T*)outbuf.ptr;
    float* bpass_arr   = (float*)bpassbuf.ptr;
    float* chanwts_arr = (float*)chanwtsbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        double zerodm = 0.0;
        for (int jj = 0; jj < nchans; jj++) {
            zerodm += indata[(nchans * ii) + jj];
        }
        for (int jj = 0; jj < nchans; jj++) {
            if (std::is_same<T, uint8_t*>::value) {
                zerodm = zerodm + 0.5;
            }
            outdata[(nchans * ii) + jj] =
                (T)((indata[(nchans * ii) + jj] - zerodm * chanwts_arr[jj]) +
                    bpass_arr[jj]);
        }
    }
}



PYBIND11_MODULE(libSigPyProc, m) {
    m.doc() = "libSigPyProc functions";

    m.def("_omp_get_max_threads", 
        []() { return omp_get_max_threads(); });
    m.def("_omp_set_num_threads", 
        [](int nthread) { omp_set_num_threads(nthread); });

    m.def("unpack", &unpack, 
        "Unpack 1, 2 and 4 bit data into an 8-bit numpy array",
        py::arg("inarray"), py::arg("nbits"));

    m.def("unpackInPlace", &unpackInPlace, 
        "Unpack 1, 2 and 4 bit data into 8-bit in the same numpy array",
        py::arg("inarray"), py::arg("nbits"));
    
    m.def("pack", &pack, 
        "Pack 1, 2 and 4 bit data into an 8-bit numpy array",
        py::arg("inarray"), py::arg("nbits"));

    m.def("packInPlace", &packInPlace, 
        "Pack 1, 2 and 4 bit data into 8-bit in the same numpy array",
        py::arg("inarray"), py::arg("nbits"));

    m.def("to8bit", &to8bit, 
        "Convert 1, 2 and 4 bit data to 8-bit",
        py::arg("inarray"), py::arg("outarray"), py::arg("flag"), 
        py::arg("fact"), py::arg("plus"), py::arg("flagMax"), 
        py::arg("flagMin"), py::arg("nsamps"), py::arg("nchans"));

    m.def("splitToBands", &splitToBands<float>);
    m.def("splitToBands", &splitToBands<uint8_t>);
    m.def("getTim", &getTim<float>);
    m.def("getTim", &getTim<uint8_t>);
    m.def("getBpass", &getBpass<float>);
    m.def("getBpass", &getBpass<uint8_t>);
    m.def("dedisperse", &dedisperse<float>);
    m.def("dedisperse", &dedisperse<uint8_t>);
    m.def("maskChannels", &maskChannels<float>);
    m.def("maskChannels", &maskChannels<uint8_t>);
    m.def("subband", &subband<float>);
    m.def("subband", &subband<uint8_t>);
    m.def("getChan", &getChan<float>);
    m.def("getChan", &getChan<uint8_t>);
    m.def("splitToChans", &splitToChans<float>);
    m.def("splitToChans", &splitToChans<uint8_t>);
    m.def("invertFreq", &invertFreq<float>);
    m.def("invertFreq", &invertFreq<uint8_t>);
    m.def("foldFil", &foldFil<float>);
    m.def("foldFil", &foldFil<uint8_t>);
    m.def("getStats", &getStats<float>);
    m.def("getStats", &getStats<uint8_t>);
    m.def("removeBandpass", &removeBandpass<float>);
    m.def("removeBandpass", &removeBandpass<uint8_t>);
    m.def("downsample", &downsample<float>);
    m.def("downsample", &downsample<uint8_t>);
    m.def("removeZeroDM", &removeZeroDM<float>);
    m.def("removeZeroDM", &removeZeroDM<uint8_t>);
}
