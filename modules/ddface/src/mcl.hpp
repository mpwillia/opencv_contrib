#ifndef __MCL_HPP
#define __MCL_HPP

#include "precomp.hpp"

namespace cv { namespace mcl {

    /*
     * e - expansion power
     * r - inflation power, granularity - higher = finer, lower = coarser
     * prune-min - drops values less than or equal to this value down to zero
     */

    void converge(Mat &mclmat, int e, double r, double prune_min); // Performs MCL iterations until convergence
    void cluster(Mat &mclmat, int iters, int e, double r, double prune_min); // Perfoms either <iters> MCL iterations or iterates until convergence

}}

#endif
