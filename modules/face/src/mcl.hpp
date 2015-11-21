#ifndef __MCL_HPP
#define __OPENCV_MCL_HPP

#include "precomp.hpp"

namespace cv { namespace mcl {

    /*
    //void mcl_normalize(Mat &src);
    //void mcl_expand(Mat &src, unsigned int e);
    void mcl_inflate(Mat &src, double power);
    void mcl_prune(Mat &src, double min);
    void mcl_iteration(Mat &src, int e, double r, double prune); // Performs one MCL iteration of expand -> inflate -> prune
    */
    
    void mcl_normalize(Mat &src);

    void converge(Mat &src, int e, double r, double prune); // Performs MCL iterations until convergence
    void cluster(Mat &src, int iters, int e, double r, double prune); // Perfoms either <iters> MCL iterations or iterates until convergence

}}

#endif
