#ifndef __MCL_HPP
#define __MCL_HPP

#include "precomp.hpp"

namespace cv { namespace mcl {

    /*
    //void mcl_normalize(Mat &src);
    //void mcl_expand(Mat &src, unsigned int e);
    void mcl_inflate(Mat &src, double power);
    void mcl_prune(Mat &src, double min);
    void mcl_iteration(Mat &src, int e, double r, double prune); // Performs one MCL iteration of expand -> inflate -> prune
    */
    
    //void normalize(Mat &src);

    void converge(Mat &mclmat, int e, double r, double prune_min); // Performs MCL iterations until convergence
    void cluster(Mat &mclmat, int iters, int e, double r, double prune_min); // Perfoms either <iters> MCL iterations or iterates until convergence

}}

#endif
