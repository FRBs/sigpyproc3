#include <cmath>
#include <omp.h>
#include <fftw3.h>
#include <complex.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <libUtils.hpp>

namespace py = pybind11;


void runningMedian(py::array_t<float> inarray, py::array_t<float> outarray,
    int window, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    Mediator* m = MediatorNew(window);
    for (int ii = 0; ii < nsamps; ii++) {
        MediatorInsert(m, indata[ii]);
        outdata[ii] = indata[ii] - (float)MediatorMedian(m);
    }
}


void runningMean(py::array_t<float> inarray, py::array_t<float> outarray,
    int window, int nsamps){
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    double sum = 0;
    for (int ii = 0; ii < nsamps; ii++) {
        sum += indata[ii];
        if (ii < window){
            outdata[ii] = indata[ii] - (float)sum / (ii + 1);
        }
        else {
            outdata[ii] = indata[ii] - (float)sum / (window + 1);
            sum -= indata[ii - window];
        }
    }
}

void runBoxcar(py::array_t<float> inarray, py::array_t<float> outarray,
    int window, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    double sum = 0;
    for (int ii = 0; ii < window; ii++) {
        sum += indata[ii];
        outdata[ii] = sum / (ii + 1);
    }

    for (int ii = window / 2; ii < nsamps - window / 2; ii++) {
        sum += indata[ii + window / 2];
        sum -= indata[ii - window / 2];
        outdata[ii] = sum / window;
    }

    for (int ii = nsamps - window; ii < nsamps; ii++) {
        outdata[ii] = sum / (nsamps - ii);
        sum -= indata[ii];
    }
}

void downsampleTim(py::array_t<float> inarray, py::array_t<float> outarray,
    int factor, int newLen) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

#pragma omp parallel for default(shared)
    for (int ii = 0; ii < newLen; ii++) {
        for (int jj = 0; jj < factor; jj++)
            outdata[ii] += indata[(ii * factor) + jj];
    }
}

void foldTim(py::array_t<float> inarray, py::array_t<double> foldarray,
             py::array_t<int32_t> countarray, double tsamp, double period,
             double accel, int nsamps, int nbins, int nints) {
    py::buffer_info inbuf = inarray.request(), foldbuf = foldarray.request();
    py::buffer_info countbuf = countarray.request();

    float*   indata    = (float*)inbuf.ptr;
    double*  fold_arr  = (double*)foldbuf.ptr;
    int32_t* count_arr = (int32_t*)countbuf.ptr;

    int   phasebin, subbint, factor1;
    float tobs, tj;
    float c = 299792458.0;

    tobs    = nsamps * tsamp;
    factor1 = (int)((nsamps / nints) + 1);

    for (int ii = 0; ii < nsamps; ii++) {
        tj       = ii * tsamp;
        phasebin = abs(((int)(nbins * tj * (1 + accel * (tj - tobs) / (2 * c)) /
                            period + 0.5))) % nbins;
        subbint = (int)ii / factor1;
        fold_arr[(subbint * nbins) + phasebin] += indata[ii];
        count_arr[(subbint * nbins) + phasebin]++;
    }
}

void rfft(py::array_t<float> inarray, py::array_t<float> outarray, int size) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    fftwf_plan plan;
    plan = fftwf_plan_dft_r2c_1d(size,
                                 indata,
                                 (fftwf_complex*)outdata,
                                 FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}

void resample(py::array_t<float> inarray, py::array_t<float> outarray,
    int nsamps, float accel, float tsamp) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    int   nsamps_by_2  = nsamps / 2;
    float partial_calc = (accel * tsamp) / (2 * 299792458.0);
    float tot_drift    = partial_calc * pow(nsamps_by_2, 2);
    int   last_bin     = 0;
    for (int ii = 0; ii < nsamps; ii++) {
        int index = ii + partial_calc * pow(ii - nsamps_by_2, 2) - tot_drift;
        outdata[index] = indata[ii];
        if (index - last_bin > 1)
            outdata[index - 1] = indata[ii];
        last_bin = index;
    }
}

PYBIND11_MODULE(libSigPyProcTim, m) {
    m.doc() = "libSigPyProcTim functions";
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);

    m.def("runningMedian", &runningMedian);
    m.def("runningMean", &runningMean);
    m.def("runBoxcar", &runBoxcar);
    m.def("foldTim", &foldTim);
    m.def("rfft", &rfft);
    m.def("resample", &resample);
}
