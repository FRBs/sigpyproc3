#pragma once

#include <cmath>
#include <omp.h>

#include "stats.hpp"

/*----------------------------------------------------------------------------*/

namespace sigpyproc {

template <typename T>
void running_median(T* inbuffer, T* outbuffer, int window, int nsamps) {
    T* arrayWithBoundary = addBoundary<T>(inbuffer, window, nsamps);
    int outSize          = nsamps + (window / 2) * 2;

    // hack for even window size
    outSize = (window % 2) ? outSize : outSize - 1;

    // Move window through all elements of the extended array
    RunningMedian<T> rm(window);
    for (int ii = 0; ii < outSize; ++ii) {
        rm.insert(arrayWithBoundary[ii]);
        if (ii >= (window - 1)) {
            outbuffer[ii - window + 1] = rm.median();
        }
    }
    // Free memory
    delete[] arrayWithBoundary;
}

template <class T>
void running_mean(T* inbuffer, float* outbuffer, int window, int nsamps) {
    T* arrayWithBoundary = addBoundary<T>(inbuffer, window, nsamps);
    int outSize          = nsamps + (window / 2) * 2;

    // hack for even window size
    outSize = (window % 2) ? outSize : outSize - 1;

    // Move window through all elements of the extended array
    double sum = 0;
    for (int ii = 0; ii < outSize; ++ii) {
        sum += arrayWithBoundary[ii];
        if (ii >= window) {
            sum -= arrayWithBoundary[ii - window];
        }
        if (ii >= (window - 1)) {
            outbuffer[ii - window + 1] = (float)sum / window;
        }
    }
    // Free memory
    delete[] arrayWithBoundary;
}

void run_boxcar(float* inbuffer, float* outbuffer, int window, int nsamps) {
    double sum = 0;
    for (int ii = 0; ii < window; ii++) {
        sum += inbuffer[ii];
        outbuffer[ii] = sum / (ii + 1);
    }

    for (int ii = window / 2; ii < nsamps - window / 2; ii++) {
        sum += inbuffer[ii + window / 2];
        sum -= inbuffer[ii - window / 2];
        outbuffer[ii] = sum / window;
    }

    for (int ii = nsamps - window; ii < nsamps; ii++) {
        outbuffer[ii] = sum / (nsamps - ii);
        sum -= inbuffer[ii];
    }
}

void downsample_tim(float* inbuffer, float* outbuffer, int factor, int newLen) {
#pragma omp parallel for default(shared)
    for (int ii = 0; ii < newLen; ii++) {
        for (int jj = 0; jj < factor; jj++)
            outbuffer[ii] += inbuffer[(ii * factor) + jj];
    }
}

void fold_tim(float* buffer, double* foldbuffer, int32_t* counts, double tsamp,
             double period, double accel, int nsamps, int nbins, int nints) {
    int phasebin, subbint, factor1;
    float tobs, tj;
    float c = 299792458.0;

    tobs    = nsamps * tsamp;
    factor1 = (int)((nsamps / nints) + 1);

    for (int ii = 0; ii < nsamps; ii++) {
        tj       = ii * tsamp;
        phasebin = abs(((int)(nbins * tj * (1 + accel * (tj - tobs) / (2 * c))
                                  / period
                              + 0.5)))
                   % nbins;
        subbint = (int)ii / factor1;
        foldbuffer[(subbint * nbins) + phasebin] += buffer[ii];
        counts[(subbint * nbins) + phasebin]++;
    }
}


void resample(float* inbuffer, float* outbuffer, int nsamps, float accel,
              float tsamp) {
    int nsamps_by_2    = nsamps / 2;
    float partial_calc = (accel * tsamp) / (2 * 299792458.0);
    float tot_drift    = partial_calc * pow(nsamps_by_2, 2);
    int last_bin       = 0;
    for (int ii = 0; ii < nsamps; ii++) {
        int index = ii + partial_calc * pow(ii - nsamps_by_2, 2) - tot_drift;
        outbuffer[index] = inbuffer[ii];
        if (index - last_bin > 1)
            outbuffer[index - 1] = inbuffer[ii];
        last_bin = index;
    }
}

}  // namespace sigpyproc