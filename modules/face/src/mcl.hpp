#ifndef __MCL_HPP
#define __OPENCV_MCL_HPP

#include "precomp.hpp"

#include <cmath>

namespace cv { namespace mcl {

    /*
    //void mcl_normalize(Mat &src);
    //void mcl_expand(Mat &src, unsigned int e);
    void mcl_inflate(Mat &src, double power);
    void mcl_prune(Mat &src, double min);
    void mcl_iteration(Mat &src, int e, double r, double prune); // Performs one MCL iteration of expand -> inflate -> prune
    void mcl_converge(Mat &src, int e, double r, double prune); // Performs MCL iterations until convergence
    void mcl_cluster(Mat &src, int iters, int e, double r, double prune); // Perfoms either <iters> MCL iterations or iterates until convergence
    */

    // Column Normalization
    void mcl_normalize(Mat &mclmat) {
        for(int i = 0; i < mclmat.cols; i++) {
            double sum = 0; 
            for(int j = 0; j < mclmat.rows; j++) {
                sum += mclmat.at<double>(i,j);
            }
            if(sum > 0)
            {
                for(int j = 0; j < mclmat.rows; j++) {
                    mclmat.at<double>(i,j) /= sum;
                }
            }
        } 
    }


    // Expansion
    void mcl_expand(Mat &mclmat, unsigned int e) {
        switch(e) {
            case 0: mclmat = Mat::eye(mclmat.rows, mclmat.cols, mclmat.type()); break; // return identity matrix
            case 1: break; // do nothing
            case 2: mclmat = mclmat * mclmat; break;
            default:
                    Mat a = mclmat.clone();
                    while(--e > 0)
                        mclmat = mclmat * a;
                    a.release();
                    break;
        }
    }


    // Inflation
    void mcl_inflate(Mat &mclmat, double r) {
        pow(mclmat, r, mclmat);
        mcl_normalize(mclmat);
    }

    // Pruning Near Zero Values
    void mcl_prune(Mat &mclmat, double min) {
        printf("\t\t\t\t - pruning...\n");
        Mat mask = (mclmat >= min) / 255; 
        mask.convertTo(mask, mclmat.type());
        mclmat = mclmat.mul(mask);
    }


    // Performs one MCL iterations
    void mcl_iteration(Mat &mclmat, int e, double r, double prune) {
        mcl_expand(mclmat, e);
        mcl_inflate(mclmat, r);
        mcl_prune(mclmat, prune);
    }
}}


#endif
