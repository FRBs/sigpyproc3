#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <complex.h>

#include <libUtils.hpp>

void ccfft(py::array_t<float> inarray, py::array_t<float> outarray, int size) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    fftwf_plan    plan;
    plan = fftwf_plan_dft_1d(size,
                             (fftwf_complex*)indata,
                             (fftwf_complex*)outdata,
                             FFTW_BACKWARD,
                             FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void ifft(py::array_t<float> inarray, py::array_t<float> outarray, int size) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    fftwf_plan plan;
    plan = fftwf_plan_dft_c2r_1d(size,
                                 (fftwf_complex*)indata,
                                 outdata,
                                 FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void formSpecInterpolated(py::array_t<float> fftarray, py::array_t<float> specarray, 
    int nsamps) {
    py::buffer_info fftbuf = fftarray.request(), specbuf = specarray.request();

    float* fft_arr  = (float*)fftbuf.ptr;
    float* spec_arr = (float*)specbuf.ptr;

    float i, r, a, b;
    float rl = 0.0, il = 0.0;
    for (int ii = 0; ii < nsamps; ii++) {
        r = fft_arr[2 * ii];
        i = fft_arr[2 * ii + 1];
        a = pow(r, 2) + pow(i, 2);
        b = (pow((r - rl), 2) + pow((i - il), 2)) / 2.;

        spec_arr[ii] = sqrt(fmax(a, b));

        rl = r;
        il = i;
    }
}

void formSpec(py::array_t<float> fftarray, py::array_t<float> specarray, int points) {
    py::buffer_info fftbuf = fftarray.request(), specbuf = specarray.request();

    float* fft_arr  = (float*)fftbuf.ptr;
    float* spec_arr = (float*)specbuf.ptr;

    float i, r;
    for (int ii = 0; ii < points; ii += 2) {
        r = fft_arr[ii];
        i = fft_arr[ii + 1];

        spec_arr[ii / 2] = sqrt(pow(r, 2) + pow(i, 2));
    }
}

void rednoise(py::array_t<float> fftarray, py::array_t<float> outarray,
    py::array_t<float> oldinarray, py::array_t<float> newinarray,
    py::array_t<float> realarray, int nsamps, float tsamp, int startwidth, 
    int endwidth, float endfreq) {
    py::buffer_info fftbuf = fftarray.request(), outbuf = outarray.request(),
                    oldinbuf = oldinarray.request(), newinbuf = newinarray.request(),
                    realbuf = realarray.request();

    float* fftdata   = (float*)fftbuf.ptr;
    float* outdata   = (float*)outbuf.ptr;
    float* oldin_arr = (float*)oldinbuf.ptr;
    float* newin_arr = (float*)newinbuf.ptr;
    float* real_arr  = (float*)realbuf.ptr;

    int   binnum  = 1;
    int   bufflen = startwidth;
    int   rindex, windex;
    int   numread_new, numread_old;
    float slope, mean_new, mean_old;
    float T = nsamps * tsamp;

    // Set DC bin to 1.0
    outdata[0] = 1.0;
    outdata[1] = 0.0;
    windex     = 2;
    rindex     = 2;

    // transfer bufflen complex samples to oldin_arr
    for (int ii = 0; ii < 2 * bufflen; ii++)
        oldin_arr[ii] = fftdata[ii + rindex];
    numread_old = bufflen;
    rindex += 2 * bufflen;

    // calculate powers for oldin_arr
    for (int ii = 0; ii < numread_old; ii++) {
        real_arr[ii] = 0;
        real_arr[ii] = oldin_arr[ii * 2] * oldin_arr[ii * 2] +
                         oldin_arr[ii * 2 + 1] * oldin_arr[ii * 2 + 1];
    }

    // calculate first median of our data and determine next bufflen
    mean_old = median(real_arr, numread_old) / log(2.0);
    binnum  += numread_old;
    bufflen  = startwidth * log(binnum);

    while (rindex / 2 < nsamps) {
        if (bufflen > nsamps - rindex / 2)
            numread_new = nsamps - rindex / 2;
        else
            numread_new = bufflen;

        for (int ii = 0; ii < 2 * numread_new; ii++)
            newin_arr[ii] = fftdata[ii + rindex];
        rindex += 2 * numread_new;

        for (int ii = 0; ii < numread_new; ii++) {
            real_arr[ii] = 0;
            real_arr[ii] = newin_arr[ii * 2] * newin_arr[ii * 2] +
                             newin_arr[ii * 2 + 1] * newin_arr[ii * 2 + 1];
        }

        mean_new = median(real_arr, numread_new) / log(2.0);
        slope    = (mean_new - mean_old) / (numread_old + numread_new);

        for (int ii = 0; ii < numread_old; ii++) {
            outdata[ii * 2 + windex]     = 0.0;
            outdata[ii * 2 + 1 + windex] = 0.0;
            outdata[ii * 2 + windex] =
                oldin_arr[ii * 2] /
                sqrt(mean_old +
                     slope * ((numread_old + numread_new) / 2.0 - ii));
            outdata[ii * 2 + 1 + windex] =
                oldin_arr[ii * 2 + 1] /
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
            oldin_arr[ii] = 0;
            oldin_arr[ii] = newin_arr[ii];
        }
    }
    for (int ii = 0; ii < 2 * numread_old; ii++) {
        outdata[ii + windex] = oldin_arr[ii] / sqrt(mean_old);
    }
}

void conjugate(py::array_t<float> specarray, py::array_t<float> outarray, int size) {
    int out_size = 2 * size - 2;
    py::buffer_info specbuf = specarray.request(), outbuf = outarray.request();

    float* spec_arr  = (float*)specbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    std::memcpy(outdata, spec_arr, size * sizeof(float));
    for (int ii = 0; ii < size - 2; ii += 2) {
        outdata[out_size - 1 - ii] = -1.0 * spec_arr[ii + 1];
        outdata[out_size - 2 - ii] = spec_arr[ii];
    }
}

void sumHarms(py::array_t<float> specarray, py::array_t<float> sumarray,
    py::array_t<int32_t> harmarray, py::array_t<int32_t> factarray, 
    int nharms, int nsamps, int nfoldi) {
    py::buffer_info specbuf = specarray.request(), sumbuf = sumarray.request(),
                    harmbuf = harmarray.request(), factbuf = factarray.request();

    float*   spec_arr = (float*)specbuf.ptr;
    float*   sum_arr  = (float*)sumbuf.ptr;
    int32_t* harm_arr = (int32_t*)harmbuf.ptr;
    int32_t* fact_arr = (int32_t*)factbuf.ptr;

    for (int ii = nfoldi; ii < nsamps - (nharms - 1); ii += nharms) {
        for (int jj = 0; jj < nharms; jj++) {
            for (int kk = 0; kk < nharms / 2; kk++) {
                sum_arr[ii + jj] +=
                    spec_arr[fact_arr[kk] + harm_arr[jj * nharms / 2 + kk]];
            }
        }
        for (int kk = 0; kk < nharms / 2; kk++) {
            fact_arr[kk] += 2 * kk + 1;
        }
    }
}

void multiply_fs(py::array_t<float> inarray, py::array_t<float> otherarray,
    py::array_t<float> outarray, int size) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();
    py::buffer_info otherbuf = otherarray.request()

    float* indata    = (float*)inbuf.ptr;
    float* outdata   = (float*)outbuf.ptr;
    float* otherdata = (float*)otherbuf.ptr;

    float sr, si, orr, oi;
    for (int ii = 0; ii < size; ii += 2) {
        sr  = indata[ii];
        si  = indata[ii + 1];
        orr = otherdata[ii];
        oi  = otherdata[ii + 1];

        outdata[ii]     = sr * orr -si * oi;
        outdata[ii + 1] = sr * oi + si * orr ;
    }
}
