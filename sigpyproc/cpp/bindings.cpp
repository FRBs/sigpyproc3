#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "data_unpack.hpp"
#include "kernels.hpp"
#include "timeseries.hpp"
#include "fourierseries.hpp"
#include "stats.hpp"

namespace py = pybind11;

/*----------------------------------------------------------------------------*/

py::array_t<uint8_t> unpack(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf = inarray.request();
    int nbytes            = inbuf.size;

    auto outarray          = py::array_t<uint8_t>(inbuf.size * 8 / nbits);
    py::buffer_info outbuf = outarray.request();

    uint8_t* indata  = (uint8_t*)inbuf.ptr;
    uint8_t* outdata = (uint8_t*)outbuf.ptr;

    sigpyproc::unpack(indata, outdata, nbits, nbytes);
    return outarray;
}

void unpackInPlace(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf = inarray.request();
    int nbytes            = inbuf.size;

    uint8_t* buffer = (uint8_t*)inbuf.ptr;

    sigpyproc::unpackInPlace(buffer, nbits, nbytes);
}

py::array_t<uint8_t> pack(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf = inarray.request();
    int nbytes            = inbuf.size;

    auto outarray          = py::array_t<uint8_t>(inbuf.size * nbits / 8);
    py::buffer_info outbuf = outarray.request();

    uint8_t* indata  = (uint8_t*)inbuf.ptr;
    uint8_t* outdata = (uint8_t*)outbuf.ptr;

    sigpyproc::pack(indata, outdata, nbits, nbytes);
    return outarray;
}

void packInPlace(py::array_t<uint8_t> inarray, int nbits) {
    py::buffer_info inbuf = inarray.request();
    int nbytes            = inbuf.size;

    uint8_t* buffer = (uint8_t*)inbuf.ptr;

    sigpyproc::packInPlace(buffer, nbits, nbytes);
}

void to_8bit(py::array_t<float> inarray, py::array_t<uint8_t> outarray,
             py::array_t<uint8_t> flag, py::array_t<float> fact,
             py::array_t<float> plus, py::array_t<float> flagMax,
             py::array_t<float> flagMin, int nsamps, int nchans) {

    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    flagbuf = flag.request(), factbuf = fact.request(),
                    plusbuf = plus.request(), flagMaxbuf = flagMax.request(),
                    flagMinbuf = flagMin.request();

    float* indata      = (float*)inbuf.ptr;
    uint8_t* outdata   = (uint8_t*)outbuf.ptr;
    uint8_t* flag_arr  = (uint8_t*)flagbuf.ptr;
    float* fact_arr    = (float*)factbuf.ptr;
    float* plus_arr    = (float*)plusbuf.ptr;
    float* flagMax_arr = (float*)flagMaxbuf.ptr;
    float* flagMin_arr = (float*)flagMinbuf.ptr;

    sigpyproc::to_8bit(indata, outdata, flag_arr, fact_arr, plus_arr,
                       flagMax_arr, flagMin_arr, nsamps, nchans);
}

template <class T>
void get_tim(py::array_t<T> inarray, py::array_t<float> outarray, int nchans,
             int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata      = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::get_tim(indata, outdata, nchans, nsamps, index);
}

template <class T>
void get_bpass(py::array_t<T> inarray, py::array_t<double> outarray, int nchans,
               int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata       = (T*)inbuf.ptr;
    double* outdata = (double*)outbuf.ptr;

    sigpyproc::get_bpass(indata, outdata, nchans, nsamps);
}

template <class T>
void dedisperse(py::array_t<T> inarray, py::array_t<float> outarray,
                py::array_t<int32_t> delays, int maxdelay, int nchans,
                int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();
    py::buffer_info delaysbuf = delays.request();

    T* indata           = (T*)inbuf.ptr;
    float* outdata      = (float*)outbuf.ptr;
    int32_t* delays_arr = (int32_t*)delaysbuf.ptr;

    sigpyproc::dedisperse(indata, outdata, delays_arr, maxdelay, nchans, nsamps,
                          index);
}

template <class T>
void mask_channels(py::array_t<T> inarray, py::array_t<uint8_t> mask,
                   int nchans, int nsamps) {
    py::buffer_info inbuf = inarray.request(), maskbuf = mask.request();

    T* indata         = (T*)inbuf.ptr;
    uint8_t* mask_arr = (uint8_t*)maskbuf.ptr;

    sigpyproc::mask_channels(indata, mask_arr, nchans, nsamps);
}

template <class T>
void subband(py::array_t<T> inarray, py::array_t<float> outarray,
             py::array_t<int32_t> delays, py::array_t<int32_t> c_to_s,
             int maxdelay, int nchans, int nsubs, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();
    py::buffer_info delaysbuf = delays.request();
    py::buffer_info c_to_sbuf = c_to_s.request();

    T* indata           = (T*)inbuf.ptr;
    float* outdata      = (float*)outbuf.ptr;
    int32_t* delays_arr = (int32_t*)delaysbuf.ptr;
    int32_t* c_to_s_arr = (int32_t*)c_to_sbuf.ptr;

    sigpyproc::subband(indata, outdata, delays_arr, c_to_s_arr, maxdelay,
                       nchans, nsubs, nsamps);
}

template <class T>
void get_chan(py::array_t<T> inarray, py::array_t<float> outarray, int chan,
              int nchans, int nsamps, int index) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata      = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::get_chan(indata, outdata, chan, nchans, nsamps, index);
}

template <class T>
void splitToChans(py::array_t<T> inarray, py::array_t<float> outarray,
                  int nchans, int nsamps, int gulp) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata      = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::splitToChans(indata, outdata, nchans, nsamps, gulp);
}

template <class T>
void invert_freq(py::array_t<T> inarray, py::array_t<T> outarray, int nchans,
                 int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

    sigpyproc::invert_freq(indata, outdata, nchans, nsamps);
}

template <class T>
void foldfil(py::array_t<T> inarray, py::array_t<float> foldarray,
             py::array_t<int32_t> countarray, py::array_t<int32_t> delays,
             int maxDelay, double tsamp, double period, double accel,
             int totnsamps, int nsamps, int nchans, int nbins, int nints,
             int nsubs, int index) {
    py::buffer_info inbuf = inarray.request(), foldbuf = foldarray.request();
    py::buffer_info countbuf  = countarray.request();
    py::buffer_info delaysbuf = delays.request();

    T* indata          = (T*)inbuf.ptr;
    float* fold_arr    = (float*)foldbuf.ptr;
    int32_t* count_arr = (int32_t*)countbuf.ptr;
    int32_t* delay_arr = (int32_t*)delaysbuf.ptr;

    sigpyproc::foldfil(indata, fold_arr, count_arr, delay_arr, maxDelay, tsamp,
                       period, accel, totnsamps, nsamps, nchans, nbins, nints,
                       nsubs, index);
}

template <class T>
void compute_moments(py::array_t<T> inarray, py::array_t<float> M1,
                     py::array_t<float> M2, py::array_t<float> M3,
                     py::array_t<float> M4, py::array_t<float> maxima,
                     py::array_t<float> minima, py::array_t<int64_t> count,
                     int nchans, int nsamps, int startflag) {

    py::buffer_info inbuf = inarray.request(), M1buf = M1.request(),
                    M2buf = M2.request(), M3buf = M3.request(),
                    M4buf = M4.request(), maxbuf = maxima.request(),
                    minbuf = minima.request(), countbuf = count.request();

    T* indata          = (T*)inbuf.ptr;
    float* M1_arr      = (float*)M1buf.ptr;
    float* M2_arr      = (float*)M2buf.ptr;
    float* M3_arr      = (float*)M3buf.ptr;
    float* M4_arr      = (float*)M4buf.ptr;
    float* max_arr     = (float*)maxbuf.ptr;
    float* min_arr     = (float*)minbuf.ptr;
    int64_t* count_arr = (int64_t*)countbuf.ptr;

    sigpyproc::compute_moments(indata, M1_arr, M2_arr, M3_arr, M4_arr, max_arr,
                               min_arr, count_arr, nchans, nsamps, startflag);
}

template <class T>
void compute_moments_simple(py::array_t<T> inarray, py::array_t<float> M1,
                            py::array_t<float> M2, py::array_t<float> maxima,
                            py::array_t<float> minima,
                            py::array_t<int64_t> count, int nchans, int nsamps,
                            int startflag) {

    py::buffer_info inbuf = inarray.request(), M1buf = M1.request(),
                    M2buf = M2.request(), maxbuf = maxima.request(),
                    minbuf = minima.request(), countbuf = count.request();

    T* indata          = (T*)inbuf.ptr;
    float* M1_arr      = (float*)M1buf.ptr;
    float* M2_arr      = (float*)M2buf.ptr;
    float* max_arr     = (float*)maxbuf.ptr;
    float* min_arr     = (float*)minbuf.ptr;
    int64_t* count_arr = (int64_t*)countbuf.ptr;

    sigpyproc::compute_moments_simple(indata, M1_arr, M2_arr, max_arr, min_arr,
                                      count_arr, nchans, nsamps, startflag);
}

template <class T>
void remove_bandpass(py::array_t<T> inarray, py::array_t<T> outarray,
                     py::array_t<float> means, py::array_t<float> stdevs,
                     int nchans, int nsamps) {

    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    meansbuf = means.request(), stdevsbuf = stdevs.request();

    T* indata         = (T*)inbuf.ptr;
    T* outdata        = (T*)outbuf.ptr;
    float* means_arr  = (float*)meansbuf.ptr;
    float* stdevs_arr = (float*)stdevsbuf.ptr;

    sigpyproc::remove_bandpass(indata, outdata, means_arr, stdevs_arr, nchans,
                               nsamps);
}

template <class T>
void downsample(py::array_t<T> inarray, py::array_t<T> outarray, int tfactor,
                int ffactor, int nchans, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

    sigpyproc::downsample(indata, outdata, tfactor, ffactor, nchans, nsamps);
}

template <class T>
void remove_zerodm(py::array_t<T> inarray, py::array_t<T> outarray,
                   py::array_t<float> bpass, py::array_t<float> chanwts,
                   int nchans, int nsamps) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request(),
                    bpassbuf = bpass.request(), chanwtsbuf = chanwts.request();

    T* indata          = (T*)inbuf.ptr;
    T* outdata         = (T*)outbuf.ptr;
    float* bpass_arr   = (float*)bpassbuf.ptr;
    float* chanwts_arr = (float*)chanwtsbuf.ptr;

    sigpyproc::remove_zerodm(indata, outdata, bpass_arr, chanwts_arr, nchans,
                             nsamps);
}

py::array_t<float> form_spec(py::array_t<float> fftarray, int specsize,
                             bool interpolated) {
    py::buffer_info fftbuf = fftarray.request();

    auto specarray          = py::array_t<float>(specsize);
    py::buffer_info specbuf = specarray.request();

    float* fft_arr  = (float*)fftbuf.ptr;
    float* spec_arr = (float*)specbuf.ptr;

    if (interpolated) {
        sigpyproc::form_spec_interpolated(fft_arr, spec_arr, specsize);
    } else {
        sigpyproc::form_spec(fft_arr, spec_arr, specsize);
    }
    return specarray;
}

py::array_t<float>
rednoise(py::array_t<float> fftarray, py::array_t<float> oldinarray,
         py::array_t<float> newinarray, py::array_t<float> realarray,
         int nsamps, float tsamp, int startwidth, int endwidth, float endfreq) {
    py::buffer_info fftbuf   = fftarray.request(),
                    oldinbuf = oldinarray.request(),
                    newinbuf = newinarray.request(),
                    realbuf  = realarray.request();

    auto outarray          = py::array_t<float>(fftbuf.size);
    py::buffer_info outbuf = outarray.request();

    float* fftdata   = (float*)fftbuf.ptr;
    float* outdata   = (float*)outbuf.ptr;
    float* oldin_arr = (float*)oldinbuf.ptr;
    float* newin_arr = (float*)newinbuf.ptr;
    float* real_arr  = (float*)realbuf.ptr;

    sigpyproc::rednoise(fftdata, outdata, oldin_arr, newin_arr, real_arr,
                        nsamps, tsamp, startwidth, endwidth, endfreq);
    return outarray;
}

py::array_t<float> conjugate(py::array_t<float> specarray, int size) {
    py::buffer_info specbuf = specarray.request();

    auto outarray          = py::array_t<float>(2 * specbuf.size - 2);
    py::buffer_info outbuf = outarray.request();

    float* spec_arr = (float*)specbuf.ptr;
    float* outdata  = (float*)outbuf.ptr;

    sigpyproc::conjugate(spec_arr, outdata, size);
    return outarray;
}

void sum_harms(py::array_t<float> specarray, py::array_t<float> sumarray,
               py::array_t<int32_t> harmarray, py::array_t<int32_t> factarray,
               int nharms, int nsamps, int nfoldi) {
    py::buffer_info specbuf = specarray.request(), sumbuf = sumarray.request(),
                    harmbuf = harmarray.request(),
                    factbuf = factarray.request();

    float* spec_arr   = (float*)specbuf.ptr;
    float* sum_arr    = (float*)sumbuf.ptr;
    int32_t* harm_arr = (int32_t*)harmbuf.ptr;
    int32_t* fact_arr = (int32_t*)factbuf.ptr;

    sigpyproc::sum_harms(spec_arr, sum_arr, harm_arr, fact_arr, nharms, nsamps,
                         nfoldi);
}

py::array_t<float> multiply_fs(py::array_t<float> inarray,
                               py::array_t<float> otherarray, int size) {
    py::buffer_info inbuf    = inarray.request();
    py::buffer_info otherbuf = otherarray.request();

    auto outarray          = py::array_t<float>(inbuf.size);
    py::buffer_info outbuf = outarray.request();

    float* indata    = (float*)inbuf.ptr;
    float* outdata   = (float*)outbuf.ptr;
    float* otherdata = (float*)otherbuf.ptr;

    sigpyproc::multiply_fs(indata, otherdata, outdata, size);
    return outarray;
}

template <class T>
py::array_t<T> running_median(py::array_t<T> inarray, int window, int nsamps) {
    py::buffer_info inbuf = inarray.request();

    auto outarray          = py::array_t<T>(nsamps);
    py::buffer_info outbuf = outarray.request();

    T* indata  = (T*)inbuf.ptr;
    T* outdata = (T*)outbuf.ptr;

    sigpyproc::running_median<T>(indata, outdata, window, nsamps);
    return outarray;
}

template <class T>
py::array_t<float> running_mean(py::array_t<T> inarray, int window,
                                int nsamps) {
    py::buffer_info inbuf = inarray.request();

    auto outarray          = py::array_t<float>(nsamps);
    py::buffer_info outbuf = outarray.request();

    T* indata      = (T*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::running_mean<T>(indata, outdata, window, nsamps);
    return outarray;
}

py::array_t<float> run_boxcar(py::array_t<float> inarray, int window,
                              int nsamps) {
    py::buffer_info inbuf = inarray.request();

    auto outarray          = py::array_t<float>(nsamps);
    py::buffer_info outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::run_boxcar(indata, outdata, window, nsamps);
    return outarray;
}

py::array_t<float> downsample_tim(py::array_t<float> inarray, int factor,
                                  int newLen) {
    py::buffer_info inbuf = inarray.request();

    auto outarray          = py::array_t<float>(newLen);
    py::buffer_info outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::downsample_tim(indata, outdata, factor, newLen);
    return outarray;
}

void fold_tim(py::array_t<float> inarray, py::array_t<double> foldarray,
              py::array_t<int32_t> countarray, double tsamp, double period,
              double accel, int nsamps, int nbins, int nints) {
    py::buffer_info inbuf = inarray.request(), foldbuf = foldarray.request();
    py::buffer_info countbuf = countarray.request();

    float* indata      = (float*)inbuf.ptr;
    double* fold_arr   = (double*)foldbuf.ptr;
    int32_t* count_arr = (int32_t*)countbuf.ptr;

    sigpyproc::fold_tim(indata, fold_arr, count_arr, tsamp, period, accel,
                        nsamps, nbins, nints);
}

void resample(py::array_t<float> inarray, py::array_t<float> outarray,
              int nsamps, float accel, float tsamp) {
    py::buffer_info inbuf = inarray.request(), outbuf = outarray.request();

    float* indata  = (float*)inbuf.ptr;
    float* outdata = (float*)outbuf.ptr;

    sigpyproc::resample(indata, outdata, nsamps, accel, tsamp);
}

PYBIND11_MODULE(libcpp, m) {
    m.doc() = "sigpyproc C++ backend.";

    m.def("omp_get_max_threads", []() { return omp_get_max_threads(); });
    m.def("omp_get_num_threads", []() { return omp_get_num_threads(); });
    m.def("omp_set_num_threads",
          [](int nthread) { omp_set_num_threads(nthread); });

    m.def("unpack", &unpack,
          "Unpack 1, 2 and 4 bit data into an 8-bit numpy array",
          py::arg("inarray"), py::arg("nbits"));

    m.def("unpackInPlace", &unpackInPlace,
          "Unpack 1, 2 and 4 bit data into 8-bit in the same numpy array",
          py::arg("inarray"), py::arg("nbits"));

    m.def("pack", &pack, "Pack 1, 2 and 4 bit data into an 8-bit numpy array",
          py::arg("inarray"), py::arg("nbits"));

    m.def("packInPlace", &packInPlace,
          "Pack 1, 2 and 4 bit data into 8-bit in the same numpy array",
          py::arg("inarray"), py::arg("nbits"));

    m.def("to_8bit", &to_8bit, "Convert 1, 2 and 4 bit data to 8-bit",
          py::arg("inarray"), py::arg("outarray"), py::arg("flag"),
          py::arg("fact"), py::arg("plus"), py::arg("flagMax"),
          py::arg("flagMin"), py::arg("nsamps"), py::arg("nchans"));

    m.def("get_tim", &get_tim<float>);
    m.def("get_tim", &get_tim<uint8_t>);
    m.def("get_bpass", &get_bpass<float>);
    m.def("get_bpass", &get_bpass<uint8_t>);
    m.def("dedisperse", &dedisperse<float>);
    m.def("dedisperse", &dedisperse<uint8_t>);
    m.def("mask_channels", &mask_channels<float>);
    m.def("mask_channels", &mask_channels<uint8_t>);
    m.def("subband", &subband<float>);
    m.def("subband", &subband<uint8_t>);
    m.def("get_chan", &get_chan<float>);
    m.def("get_chan", &get_chan<uint8_t>);
    m.def("splitToChans", &splitToChans<float>);
    m.def("splitToChans", &splitToChans<uint8_t>);
    m.def("invert_freq", &invert_freq<float>);
    m.def("invert_freq", &invert_freq<uint8_t>);
    m.def("foldfil", &foldfil<float>);
    m.def("foldfil", &foldfil<uint8_t>);
    m.def("compute_moments", &compute_moments<float>);
    m.def("compute_moments", &compute_moments<uint8_t>);
    m.def("compute_moments_simple", &compute_moments_simple<float>);
    m.def("compute_moments_simple", &compute_moments_simple<uint8_t>);
    m.def("remove_bandpass", &remove_bandpass<float>);
    m.def("remove_bandpass", &remove_bandpass<uint8_t>);
    m.def("downsample", &downsample<float>);
    m.def("downsample", &downsample<uint8_t>);
    m.def("remove_zerodm", &remove_zerodm<float>);
    m.def("remove_zerodm", &remove_zerodm<uint8_t>);

    m.def("form_spec", &form_spec);
    m.def("rednoise", &rednoise);
    m.def("conjugate", &conjugate);
    m.def("sum_harms", &sum_harms);
    m.def("multiply_fs", &multiply_fs);

    m.def("running_median", &running_median<double>);
    m.def("running_median", &running_median<float>);
    m.def("running_median", &running_median<int64_t>);
    m.def("running_median", &running_median<int32_t>);
    m.def("running_median", &running_median<int8_t>);
    m.def("running_median", &running_median<uint8_t>);
    m.def("running_mean", &running_mean<double>);
    m.def("running_mean", &running_mean<float>);
    m.def("running_mean", &running_mean<int64_t>);
    m.def("running_mean", &running_mean<int32_t>);
    m.def("running_mean", &running_mean<int8_t>);
    m.def("running_mean", &running_mean<uint8_t>);
    m.def("run_boxcar", &run_boxcar);
    m.def("downsample_tim", &downsample_tim);
    m.def("fold_tim", &fold_tim);
    m.def("resample", &resample);
}
