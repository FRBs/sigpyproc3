#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <cstdlib>
#include <queue>
#include <algorithm>

#define MACRO_STRINGIFY(x) #x

/**
 * @brief Get a random number for given statistics
 *
 * @param mean Mean of the normal distribution
 * @param std  Standard deviation of the distribution
 * @return     A generated random number
 */
unsigned char getRand(float mean, float std) {
    unsigned char randval;

    std::random_device rd;
    // Create and seed the generator (32-bit Mersenne Twister engine)
    std::mt19937 gen(rd());
    // Create distribution
    std::normal_distribution<> dist(mean, std);
    // Generate random numbers according to distribution

    randval = std::round(dist(gen));
    if (randval > 255)
        randval = 255;
    else if (randval < 0)
        randval = 0;
    return randval;
}

/**
 * @brief Find the median of a std::vector
 *
 * @tparam T Any type such as int, float or double
 * @param X  Input vector
 * @return   The median value as a type T.
 *
 * @details
 * Taken from
 * https://gitlab.com/conradsnicta/armadillo-code/-/blob/10.1.x/include/armadillo_bits/op_median_meat.hpp
 */
template <typename T>
T direct_median(std::vector<T>& X) {
    const int n_elem = int(X.size());
    const int half   = n_elem / 2;

    typename std::vector<T>::iterator first    = X.begin();
    typename std::vector<T>::iterator nth      = first + half;
    typename std::vector<T>::iterator pastlast = X.end();

    std::nth_element(first, nth, pastlast);

    if ((n_elem % 2) == 0) {
        // even number of elements
        typename std::vector<T>::iterator start   = X.begin();
        typename std::vector<T>::iterator pastend = start + half;

        const T val1 = (*nth);
        const T val2 = (*(std::max_element(start, pastend)));

        return val1 + (val2 - val1) / T(2);
    } else {
        // odd number of elements
        return (*nth);
    }
}

/**
 * @brief Find the median of an array
 *
 * @tparam T   Any type such as int, float or double
 * @param arr  Input array
 * @param n    Size of the input array
 * @return     The median value as a type T.
 *
 * @see direct_median
 */
template <typename T>
T median(T* arr, int n) {
    std::vector<T> input_vector(arr, arr + n);
    return direct_median<T>(input_vector);
}

/**
 * @brief Efficient running median calculator
 *
 * @tparam T Any type such as int, float or double
 *
 * @details
 * Based on https://github.com/thomedes/RunningMedian.cpp
 */
template <typename T>
class RunningMedian {
private:
    const size_t window_;
    // using a pool of values (sorted_) that is always kept sorted
    std::queue<T> ring_;
    std::vector<T> sorted_;

    void ring_push(T x) {
        ring_.push(x);
        if (ring_.size() > window_) {
            auto last_pos = std::lower_bound(sorted_.begin(), sorted_.end(),
                                             ring_.front());
            sorted_.erase(last_pos);
            ring_.pop();
        }
        auto insert_pos = std::lower_bound(sorted_.begin(), sorted_.end(), x);
        sorted_.insert(insert_pos, x);
    }

public:
    /**
     * @brief Construct a new Running Median object
     *
     * @param window_size  Size of the moving window
     */
    RunningMedian(size_t window_size) : window_(window_size) {}

    /**
     * @brief Push a new value into the moving window ring buffer
     *
     * @param x A new value
     */
    void insert(T x) { ring_push(x); }

    /**
     * @brief Get the mdian from the current moving window
     *
     * @return The median value as a type T.
     */
    T median() const {
        // return new median
        const size_t n = sorted_.size();  // Current window size
        const size_t m = n / 2;

        return (n % 2) ? sorted_[m] : ((sorted_[m - 1] + sorted_[m]) / 2);
    }
};

/**
 * @brief Extend the given array alongside the boundary edges
 *
 * @tparam T        Any type such as int, float or double
 * @param inbuffer  Inpuy array
 * @param window    Moving window size
 * @param nsamps    Size of the input array
 * @return          A new exxtended array as type T.
 */
template <typename T>
T* addBoundary(T* inbuffer, int window, int nsamps) {
    // Allocate memory for new extended array (to deal with window edges)
    const int boundarySize = window / 2;  // integer division
    const int outSize      = nsamps + boundarySize * 2;

    T* arrayWithBoundary = new T[outSize];
    std::memcpy(arrayWithBoundary + boundarySize, inbuffer,
                nsamps * sizeof(float));
    // Extend by reflecting about the edge.
    for (int ii = 0; ii < boundarySize; ++ii) {
        arrayWithBoundary[ii] = inbuffer[boundarySize - 1 - ii];
        arrayWithBoundary[nsamps + boundarySize + ii]
            = inbuffer[nsamps - 1 - ii];
    }
    return arrayWithBoundary;
}