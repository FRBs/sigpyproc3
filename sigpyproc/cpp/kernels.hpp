#pragma once

#include <cstring>
#include <type_traits>
#include <omp.h>

/*----------------------------------------------------------------------------*/

namespace sigpyproc {

/**
 * Convert 1-,2- or 4-bit data to 8-bit data and write to file.
 */

void to_8bit(float* inbuffer, uint8_t* outbuffer, uint8_t* flagbuffer,
             float* factbuffer, float* plusbuffer, float* flagMax,
             float* flagMin, int nsamps, int nchans) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(ii * nchans) + jj]
                = inbuffer[(ii * nchans) + jj] / factbuffer[jj]
                  - plusbuffer[jj];
            if (inbuffer[(ii * nchans) + jj] > flagMax[jj])
                flagbuffer[(ii * nchans) + jj] = 2;
            else if (inbuffer[(ii * nchans) + jj] < flagMin[jj])
                flagbuffer[(ii * nchans) + jj] = 0;
            else
                flagbuffer[(ii * nchans) + jj] = 1;
        }
    }
}

template <class T>
void get_tim(T* inbuffer, float* outbuffer, int nchans, int nsamps, int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[index + ii] += inbuffer[(nchans * ii) + jj];
        }
    }
}

template <class T>
void get_bpass(T* inbuffer, double* outbuffer, int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        for (int ii = 0; ii < nsamps; ii++) {
            outbuffer[jj] += inbuffer[(nchans * ii) + jj];
        }
    }
}

template <class T>
void dedisperse(T* inbuffer, float* outbuffer, int32_t* delays, int maxdelay,
                int nchans, int nsamps, int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[index + ii]
                += inbuffer[(ii * nchans) + (delays[jj] * nchans) + jj];
        }
    }
}

template <class T>
void mask_channels(T* inbuffer, uint8_t* mask, int nchans, int nsamps) {
    for (int ii = 0; ii < nchans; ii++) {
        if (mask[ii] == 0) {
            for (int jj = 0; jj < nsamps; jj++) {
                inbuffer[jj * nchans + ii] = 0.0;
            }
        }
    }
}

template <class T>
void subband(T* inbuffer, float* outbuffer, int32_t* delays, int32_t* c_to_s,
             int maxdelay, int nchans, int nsubs, int nsamps) {
    int out_size;
    out_size = nsubs * (nsamps - maxdelay) * sizeof(float);
    std::memset(outbuffer, 0.0, out_size);

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < (nsamps - maxdelay); ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(ii * nsubs) + c_to_s[jj]]
                += (float)inbuffer[(ii * nchans) + (delays[jj] * nchans) + jj];
        }
    }
}

template <class T>
void get_chan(T* inbuffer, float* outbuffer, int chan, int nchans, int nsamps,
              int index) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        outbuffer[index + ii] = inbuffer[(ii * nchans) + chan];
    }
}

template <class T>
void splitToChans(T* inbuffer, float* outbuffer, int nchans, int nsamps,
                  int gulp) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(jj * gulp) + ii] = inbuffer[(ii * nchans) + jj];
        }
    }
}

template <class T>
void invert_freq(T* inbuffer, T* outbuffer, int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        for (int jj = 0; jj < nchans; jj++) {
            outbuffer[(jj) + ii * nchans]
                = inbuffer[(nchans - 1 - jj) + ii * nchans];
        }
    }
}

template <class T>
void foldfil(T* inbuffer, float* foldbuffer, int32_t* countbuffer,
             int32_t* delays, int maxDelay, double tsamp, double period,
             double accel, int totnsamps, int nsamps, int nchans, int nbins,
             int nints, int nsubs, int index) {
    int tobs, phasebin, subband, subint, pos1, pos2;
    float factor1, factor2, val, tj;
    float c = 299792458.0;
    factor1 = (float)totnsamps / nints;
    factor2 = (float)nchans / nsubs;
    tobs    = (int)(totnsamps * tsamp);
    for (int ii = 0; ii < (nsamps - maxDelay); ii++) {
        tj = (ii + index) * tsamp;
        phasebin
            = ((int)(nbins * tj * (1 + accel * (tj - tobs) / (2 * c)) / period
                     + 0.5))
              % nbins;
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
 * compute_moments: Computing central moments in one pass through the data,
 * the algorithm is numerically stable and accurate.
 * Ref:
 * https://www.johndcook.com/blog/skewness_kurtosis/
 * https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
 */
template <class T>
void compute_moments(T* inbuffer, float* M1, float* M2, float* M3, float* M4,
                     float* maxbuffer, float* minbuffer, int64_t* count,
                     int nchans, int nsamps, int startflag) {
    T val;
    if (startflag == 0) {
        for (int jj = 0; jj < nchans; jj++) {
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
            M4[jj] += term1 * delta_n2 * (n * n - 3 * n + 3)
                      + 6 * delta_n2 * M2[jj] - 4 * delta_n * M3[jj];
            M3[jj] += term1 * delta_n * (n - 2) - 3 * delta_n * M2[jj];
            M2[jj] += term1;

            if (val > maxbuffer[jj])
                maxbuffer[jj] = val;
            else if (val < minbuffer[jj])
                minbuffer[jj] = val;
        }
    }
}

template <class T>
void compute_moments_simple(T* inbuffer, float* M1, float* M2, float* maxbuffer,
                            float* minbuffer, int64_t* count, int nchans,
                            int nsamps, int startflag) {
    T val;
    if (startflag == 0) {
        for (int jj = 0; jj < nchans; jj++) {
            maxbuffer[jj] = inbuffer[jj];
            minbuffer[jj] = inbuffer[jj];
        }
    }
#pragma omp parallel for default(shared)
    for (int jj = 0; jj < nchans; jj++) {
        double delta, delta_n;
        for (int ii = 0; ii < nsamps; ii++) {
            val = inbuffer[(nchans * ii) + jj];
            count[jj] += 1;
            long long n = count[jj];

            delta   = val - M1[jj];
            delta_n = delta / n;
            M1[jj] += delta_n;
            M2[jj] += delta * delta_n * (n - 1);

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
void remove_bandpass(T* inbuffer, T* outbuffer, float* means, float* stdevs,
                     int nchans, int nsamps) {
    T val;
    float DIGI_MEAN  = 127.5;
    float DIGI_SIGMA = 6;
    float DIGI_SCALE = DIGI_MEAN / DIGI_SIGMA;
    int DIGI_MIN     = 0;
    int DIGI_MAX     = 255;

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
            int pos    = nchans * ii * tfactor + jj * ffactor;
            float temp = 0;
            for (int kk = 0; kk < tfactor; kk++) {
                for (int ll = 0; ll < ffactor; ll++) {
                    temp += inbuffer[kk * nchans + ll + pos];
                }
            }
            float result = temp / totfactor;
            if (std::is_same<T, uint8_t*>::value) {
                result = result + 0.5;
            }
            outbuffer[ii * newnchans + jj] = (T)(result);
        }
    }
}

/**
 * Remove the channel-weighted zero-DM (Eatough, Keane & Lyne 2009)
 */
template <class T>
void remove_zerodm(T* inbuffer, T* outbuffer, float* bpass, float* chanwts,
                   int nchans, int nsamps) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < nsamps; ii++) {
        double zerodm = 0.0;
        for (int jj = 0; jj < nchans; jj++) {
            zerodm += inbuffer[(nchans * ii) + jj];
        }
        for (int jj = 0; jj < nchans; jj++) {
            if (std::is_same<T, uint8_t*>::value) {
                zerodm = zerodm + 0.5;
            }
            outbuffer[(nchans * ii) + jj]
                = (T)((inbuffer[(nchans * ii) + jj] - zerodm * chanwts[jj])
                      + bpass[jj]);
        }
    }
}

}  // namespace sigpyproc