#include <cstring>
#include <type_traits>
#include <omp.h>

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
void unpack(unsigned char* inbuffer, unsigned char* outbuffer, int nbits,
            int nbytes) {
    int ii, jj;
    switch (nbits) {
        case 1:
            for (ii = 0; ii < nbytes; ii++) {
                for (jj = 0; jj < 8; jj++) {
                    outbuffer[(ii * 8) + jj] = (inbuffer[ii] >> jj) & 1;
                }
            }
            break;
        case 2:
            for (ii = 0; ii < nbytes; ii++) {
                outbuffer[(ii * 4) + 3] = inbuffer[ii] & LO2BITS;
                outbuffer[(ii * 4) + 2] = (inbuffer[ii] & LOMED2BITS) >> 2;
                outbuffer[(ii * 4) + 1] = (inbuffer[ii] & UPMED2BITS) >> 4;
                outbuffer[(ii * 4) + 0] = (inbuffer[ii] & HI2BITS) >> 6;
            }
            break;
        case 4:
            for (ii = 0; ii < nbytes; ii++) {
                outbuffer[(ii * 2) + 1] = inbuffer[ii] & LO4BITS;
                outbuffer[(ii * 2) + 0] = (inbuffer[ii] & HI4BITS) >> 4;
            }
            break;
    }
}

/**
 * Function to unpack 1,2 and 4 bit data
 * Data is unpacked into the same buffer. This is done by unpacking the bytes
 * backwards so as not to overwrite any of the data. This is old code that is
 * no longer used should the filterbank reader ever be changed from using
 * np.fromfile this may once again become useful
 * Note: Only set up for big endian bit ordering
 */
void unpackInPlace(unsigned char* buffer, int nbits, int nbytes) {
    int ii, jj, pos;
    int lastsamp = nbits * nbytes / 8;

    unsigned char temp;

    switch (nbits) {
        case 1:
            for (ii = lastsamp - 1; ii > -1; ii--) {
                temp = buffer[ii];
                pos  = ii * 8;
                for (jj = 0; jj < 8; jj++) {
                    buffer[pos + jj] = (temp >> jj) & 1;
                }
            }
            break;
        case 2:
            for (ii = lastsamp - 1; ii > -1; ii--) {
                temp            = buffer[ii];
                pos             = ii * 4;
                buffer[pos + 3] = temp & LO2BITS;
                buffer[pos + 2] = (temp & LOMED2BITS) >> 2;
                buffer[pos + 1] = (temp & UPMED2BITS) >> 4;
                buffer[pos + 0] = (temp & HI2BITS) >> 6;
            }
            break;
        case 4:
            for (ii = lastsamp - 1; ii > -1; ii--) {
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
void pack(unsigned char* buffer, unsigned char* outbuffer, int nbits,
          int nbytes) {
    int ii, jj, pos;
    int times   = pow(nbits, 2);
    int bitfact = 8 / nbits;

    unsigned char val;

    switch (nbits) {
        case 1:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 8;
                val = (buffer[pos + 7] << 7) | (buffer[pos + 6] << 6) |
                      (buffer[pos + 5] << 5) | (buffer[pos + 4] << 4) |
                      (buffer[pos + 3] << 3) | (buffer[pos + 2] << 2) |
                      (buffer[pos + 1] << 1) | buffer[pos + 0];
                outbuffer[ii] = val;
            }
            break;
        case 2:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 4;
                val = (buffer[pos] << 6) | (buffer[pos + 1] << 4) |
                      (buffer[pos + 2] << 2) | buffer[pos + 3];
                outbuffer[ii] = val;
            }
            break;
        case 4:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 2;
                val = (buffer[pos] << 4) | buffer[pos + 1];

                outbuffer[ii] = val;
            }
            break;
    }
}

/**
 * Function to pack bit data into the same buffer
 */
void packInPlace(unsigned char* buffer, int nbits, int nbytes) {
    int ii, jj, pos;
    int times   = pow(nbits, 2);
    int bitfact = 8 / nbits;

    unsigned char val;

    switch (nbits) {
        case 1:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 8;
                val = (buffer[pos + 7] << 7) | (buffer[pos + 6] << 6) |
                      (buffer[pos + 5] << 5) | (buffer[pos + 4] << 4) |
                      (buffer[pos + 3] << 3) | (buffer[pos + 2] << 2) |
                      (buffer[pos + 1] << 1) | buffer[pos + 0];
                buffer[ii] = val;
            }
            break;
        case 2:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 4;
                val = (buffer[pos] << 6) | (buffer[pos + 1] << 4) |
                      (buffer[pos + 2] << 2) | buffer[pos + 3];
                buffer[ii] = val;
            }
            break;
        case 4:
            for (ii = 0; ii < nbytes / bitfact; ii++) {
                pos = ii * 2;
                val = (buffer[pos] << 4) | buffer[pos + 1];

                buffer[ii] = val;
            }
            break;
    }
}

void to8bit(float* inbuffer, unsigned char* outbuffer,
            unsigned char* flagbuffer, float* factbuffer, float* plusbuffer,
            float* flagMax, float* flagMin, int nsamps, int nchans) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(ii * nchans) + jj] =
                inbuffer[(ii * nchans) + jj] / factbuffer[jj] - plusbuffer[jj];
            if (inbuffer[(ii * nchans) + jj] > flagMax[jj])
                flagbuffer[(ii * nchans) + jj] = 2;
            else if (inbuffer[(ii * nchans) + jj] < flagMin[jj])
                flagbuffer[(ii * nchans) + jj] = 0;
            else
                flagbuffer[(ii * nchans) + jj] = 1;
        }
    }
}

unsigned char getRand(float mean, float std) {
    unsigned char randval;
    randval = (std * normDist() + mean);
    if (randval > 255)
        randval = 255;
    else if (randval < 0)
        randval = 0;
    return randval;
}

/*
TODO: This code is very slow compared to python slicing.
*/
void splitToBands(unsigned char* inbuffer, unsigned char* outbuffer, int nchans,
                  int nsamps, int nsub, int chanpersub, int chanstart) {
    int ii, jj;
    int chan_end = nsub * chanpersub + chanstart;
#pragma omp parallel for default(shared) private(ii, jj)
    for (ii = 0; ii < nsamps; ii++) {
        for (jj = chanstart; jj < chan_end; jj++) {
            outbuffer[(ii * nchans) +
                      (jj % chanpersub * nsub + jj / chanpersub)] =
                inbuffer[(ii * nchans) + jj];
        }
    }
}

template <class T>
void getTim(T* inbuffer, float* outbuffer, int nchans, int nsamps, int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[index + ii] += inbuffer[(nchans * ii) + jj];
        }
    }
}

template <class T>
void getBpass(T* inbuffer, double* outbuffer, int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        for (int ii = 0; ii < nsamps; ii++) {
            outbuffer[jj] += inbuffer[(nchans * ii) + jj];
        }
    }
}

template <class T>
void dedisperse(T* inbuffer, float* outbuffer, int* delays, int maxdelay,
                int nchans, int nsamps, int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[index + ii] +=
                inbuffer[(ii * nchans) + (delays[jj] * nchans) + jj];
        }
    }
}

template <class T>
void maskChannels(T* inbuffer, unsigned char* mask, int nchans, int nsamps) {
    for (int ii = 0; ii < nchans; ii++) {
        if (mask[ii] == 0) {
            for (int jj = 0; jj < nsamps; jj++) {
                inbuffer[jj * nchans + ii] = 0.0;
            }
        }
    }
}

template <class T>
void subband(T* inbuffer, float* outbuffer, int* delays, int* c_to_s,
             int maxdelay, int nchans, int nsubs, int nsamps) {
    int out_size;
    out_size = nsubs * (nsamps - maxdelay) * sizeof(float);
    std::memset(outbuffer, 0.0, out_size);
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(ii * nsubs) + c_to_s[jj]] +=
                (float)inbuffer[(ii * nchans) + (delays[jj] * nchans) + jj];
        }
    }
}

template <class T>
void getChan(T* inbuffer, float* outbuffer, int chan, int nchans, int nsamps,
             int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++)
        outbuffer[index + ii] = inbuffer[(ii * nchans) + chan];
}

template <class T>
void splitToChans(T* inbuffer, float* outbuffer, int nchans, int nsamps,
                  int gulp) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++)
            outbuffer[(jj * gulp) + ii] = inbuffer[(ii * nchans) + jj];
    }
}

template <class T>
void invertFreq(T* inbuffer, T* outbuffer, int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(jj) + ii * nchans] =
                inbuffer[(nchans - 1 - jj) + ii * nchans];
        }
    }
}

template <class T>
void foldFil(T* inbuffer, float* foldbuffer, int* countbuffer, int* delays,
             int maxDelay, double tsamp, double period, double accel,
             int totnsamps, int nsamps, int nchans, int nbins, int nints,
             int nsubs, int index) {
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
                   0.5)) % nbins;
        subint = (int)((index + ii) / factor1);
        pos1   = (subint * nsubs * nbins) + phasebin;
        for (int jj = 0; jj < nchans; jj++) {
            subband = (int)(jj / factor2);
            pos2    = pos1 + (subband * nbins);
            val     = inbuffer[(ii * nchans) + (delays[jj] * nchans) + jj];
            foldbuffer[pos2] += val;
            countbuffer[pos2]++;
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
void getStats(T* inbuffer, float* M1, float* M2, float* M3, float* M4,
              float* maxbuffer, float* minbuffer, long long* count, int nchans,
              int nsamps, int startflag) {
    T val;
    if (startflag == 0) {
        for (jj = 0; jj < nchans; jj++) {
            maxbuffer[jj] = inbuffer[jj];
            minbuffer[jj] = inbuffer[jj];
        }
    }
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        double delta, delta_n, delta_n2, term1;
        for (int ii = 0; ii < nsamps; ii++) {
            val = inbuffer[(nchans * ii) + jj];
            count[jj] += 1;
            long long n = count[jj];

            delta    = val - M1[jj];
            delta_n  = delta / n;
            delta_n2 = delta_n * delta_n;
            term1    = delta * delta_n * (n - 1);
            M1[jj] += delta_n;
            M4[jj] += term1 * delta_n2 * (n * n - 3 * n + 3) +
                      6 * delta_n2 * M2[jj] - 4 * delta_n * M3[jj];
            M3[jj] += term1 * delta_n * (n - 2) - 3 * delta_n * M2[jj];
            M2[jj] += term1;

            if (val > maxbuffer[jj])
                maxbuffer[jj] = val;
            else if (val < minbuffer[jj])
                minbuffer[jj] = val;
        }
    }
}

/**
 * Digitizing code taken from SigProcDigitizer.C (dspsr)
 */
template <class T>
void removeBandpass(T* inbuffer, T* outbuffer, float* means, float* stdevs,
                    int nchans, int nsamps) {
    T     val;
    float DIGI_MEAN  = 127.5;
    float DIGI_SIGMA = 6;
    float DIGI_SCALE = DIGI_MEAN / DIGI_SIGMA;
    int   DIGI_MIN   = 0;
    int   DIGI_MAX   = 255;

#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        for (int ii = 0; ii < nsamps; ii++) {
            val = inbuffer[(nchans * ii) + jj];

            double scale;
            if (stdevs[jj] == 0.0)
                scale = 1.0;
            else
                scale = 1.0 / stdevs[jj];

            // Normalize the data per channel to N(0,1)
            double normval = (val - means[jj]) * scale;

            if (std::is_same<T, unsigned char*>::value) {
                // Shift the data for digitization
                normval = ((normval * DIGI_SCALE) + DIGI_MEAN + 0.5);
                // clip the normval at the limits
                if (normval < DIGI_MIN)
                    normval = DIGI_MIN;

                if (normval > DIGI_MAX)
                    normval = DIGI_MAX;
            }
            outbuffer[(nchans * ii) + jj] = (T)normval;
        }
    }
}

template <class T>
void downsample(T* inbuffer, T* outbuffer, int tfactor, int ffactor, int nchans,
                int nsamps) {
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
                    temp += inbuffer[kk * nchans + ll + pos];
                }
            }
            float result =
                temp /
                totfactor if (std::is_same<T, unsigned char*>::value){
                    result = result + 0.5} outbuffer[ii * newnchans + jj] =
                    (T)(result);
        }
    }
}

/**
 * Remove the channel-weighted zero-DM (Eatough, Keane & Lyne 2009)
 * code based on zerodm.c (presto)
 */
template <class T>
void removeZeroDM(T* inbuffer, T* outbuffer, float* bpass, float* chanwts,
                  int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        double zerodm = 0.0;
        for (int jj = 0; jj < nchans; jj++) {
            zerodm += inbuffer[(nchans * ii) + jj];
        }
        for (jj = 0; jj < nchans; jj++) {
            if (std::is_same<T, unsigned char*>::value) {
                zerodm = zerodm + 0.5
            }
            outbuffer[(nchans * ii) + jj] =
                (T)((inbuffer[(nchans * ii) + jj] - zerodm * chanwts[jj]) +
                    bpass[jj]);
        }
    }
}
