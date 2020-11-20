#pragma once

#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <complex.h>

#include "utils.hpp"

/*----------------------------------------------------------------------------*/

namespace sigpyproc {

/**
 * Complex One-Dimensional DFTs
 */
void ccfft(float* inbuffer, float* outbuffer, int size) {
    fftwf_plan plan;
    plan = fftwf_plan_dft_1d(size,
                             (fftwf_complex*)inbuffer,
                             (fftwf_complex*)outbuffer,
                             FFTW_BACKWARD,
                             FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void ifft(float* inbuffer, float* outbuffer, int size) {
    fftwf_plan plan;
    plan = fftwf_plan_dft_c2r_1d(size,
                                 (fftwf_complex*)inbuffer,
                                 outbuffer,
                                 FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void formSpecInterpolated(float* fftbuffer, float* specbuffer, int specsize) {
    float i, r, a, b;
    float rl = 0.0, il = 0.0;
    for (int ii = 0; ii < specsize; ii++) {
        r = fftbuffer[2 * ii];
        i = fftbuffer[2 * ii + 1];
        a = pow(r, 2) + pow(i, 2);
        b = (pow((r - rl), 2) + pow((i - il), 2)) / 2.;

        specbuffer[ii] = sqrt(fmax(a, b));

        rl = r;
        il = i;
    }
}

void formSpec(float* fftbuffer, float* specbuffer, int specsize) {
    float i, r;
    for (int ii = 0; ii < specsize; ii++) {
        r = fftbuffer[2 * ii];
        i = fftbuffer[2 * ii + 1];

        specbuffer[ii] = sqrt(pow(r, 2) + pow(i, 2));
    }
}

void rednoise(float* fftbuffer, float* outbuffer, float* oldinbuf,
              float* newinbuf, float* realbuffer, int nsamps, float tsamp,
              int startwidth, int endwidth, float endfreq) {
    int   binnum  = 1;
    int   bufflen = startwidth;
    int   rindex, windex;
    int   numread_new, numread_old;
    float slope, mean_new, mean_old;
    float T = nsamps * tsamp;

    // Set DC bin to 1.0
    outbuffer[0] = 1.0;
    outbuffer[1] = 0.0;
    windex     = 2;
    rindex     = 2;

    // transfer bufflen complex samples to oldinbuf
    for (int ii = 0; ii < 2 * bufflen; ii++)
        oldinbuf[ii] = fftbuffer[ii + rindex];
    numread_old = bufflen;
    rindex += 2 * bufflen;

    // calculate powers for oldinbuf
    for (int ii = 0; ii < numread_old; ii++) {
        realbuffer[ii] = 0;
        realbuffer[ii] = oldinbuf[ii * 2] * oldinbuf[ii * 2] +
                         oldinbuf[ii * 2 + 1] * oldinbuf[ii * 2 + 1];
    }

    // calculate first median of our data and determine next bufflen
    mean_old = median<float>(realbuffer, numread_old) / log(2.0);
    binnum  += numread_old;
    bufflen  = startwidth * log(binnum);

    while (rindex / 2 < nsamps) {
        if (bufflen > nsamps - rindex / 2)
            numread_new = nsamps - rindex / 2;
        else
            numread_new = bufflen;

        for (int ii = 0; ii < 2 * numread_new; ii++)
            newinbuf[ii] = fftbuffer[ii + rindex];
        rindex += 2 * numread_new;

        for (int ii = 0; ii < numread_new; ii++) {
            realbuffer[ii] = 0;
            realbuffer[ii] = newinbuf[ii * 2] * newinbuf[ii * 2] +
                             newinbuf[ii * 2 + 1] * newinbuf[ii * 2 + 1];
        }

        mean_new = median<float>(realbuffer, numread_new) / log(2.0);
        slope    = (mean_new - mean_old) / (numread_old + numread_new);

        for (int ii = 0; ii < numread_old; ii++) {
            outbuffer[ii * 2 + windex]     = 0.0;
            outbuffer[ii * 2 + 1 + windex] = 0.0;
            outbuffer[ii * 2 + windex] =
                oldinbuf[ii * 2] /
                sqrt(mean_old +
                     slope * ((numread_old + numread_new) / 2.0 - ii));
            outbuffer[ii * 2 + 1 + windex] =
                oldinbuf[ii * 2 + 1] /
                sqrt(mean_old +
                     slope * ((numread_old + numread_new) / 2.0 - ii));
        }
        windex += 2 * numread_old;

        binnum += numread_new;
        if ((binnum * 1.0) / T < endfreq)
            bufflen = startwidth * log(binnum);
        else
            bufflen = endwidth;
        numread_old = numread_new;
        mean_old    = mean_new;

        for (int ii = 0; ii < 2 * numread_new; ii++) {
            oldinbuf[ii] = 0;
            oldinbuf[ii] = newinbuf[ii];
        }
    }
    for (int ii = 0; ii < 2 * numread_old; ii++) {
        outbuffer[ii + windex] = oldinbuf[ii] / sqrt(mean_old);
    }
}

void conjugate(float* specbuffer, float* outbuffer, int size) {
    int out_size = 2 * size - 2;
    std::memcpy(outbuffer, specbuffer, size * sizeof(float));
    for (int ii = 0; ii < size - 2; ii += 2) {
        outbuffer[out_size - 1 - ii] = -1.0 * specbuffer[ii + 1];
        outbuffer[out_size - 2 - ii] = specbuffer[ii];
    }
}

void sumHarms(float* specbuffer, float* sumbuffer, int32_t* sumarray,
              int32_t* factarray, int nharms, int nsamps, int nfoldi) {
    for (int ii = nfoldi; ii < nsamps - (nharms - 1); ii += nharms) {
        for (int jj = 0; jj < nharms; jj++) {
            for (int kk = 0; kk < nharms / 2; kk++) {
                sumbuffer[ii + jj] +=
                    specbuffer[factarray[kk] + sumarray[jj * nharms / 2 + kk]];
            }
        }
        for (int kk = 0; kk < nharms / 2; kk++) {
            factarray[kk] += 2 * kk + 1;
        }
    }
}

void multiply_fs(float* inbuffer, float* otherbuffer, float* outbuffer,
                 int size) {
    float sr, si, orr, oi;
    for (int ii = 0; ii < size; ii += 2) {
        sr  = inbuffer[ii];
        si  = inbuffer[ii + 1];
        orr = otherbuffer[ii];
        oi  = otherbuffer[ii + 1];

        outbuffer[ii]     = sr * orr -si * oi;
        outbuffer[ii + 1] = sr * oi + si * orr ;
    }
}

} // namespace sigpyproc