
#include "opencv2/face/precomp.hpp"
#include "mcl.hpp"
#include <cmath>

#define COMP_EPSILON 0.00001

namespace cv { namespace mcl {

    // Column Normalization
    void normalize(Mat &mclmat) {
        //printf("\t\t\tnormalize\n");
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
    void expand(Mat &mclmat, unsigned int e) {

        //printf("\t\t\texpand\n");
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
    void inflate(Mat &mclmat, double r) {
        //printf("\t\t\tinflate (then normalize)\n");
        pow(mclmat, r, mclmat);
        normalize(mclmat);
    }

    // Pruning Near Zero Values
    void prune(Mat &mclmat, double min) {
        //printf("\t\t\tprune\n");
        Mat mask = (mclmat >= min) / 255; 
        mask.convertTo(mask, mclmat.type());
        mclmat = mclmat.mul(mask);
    }

    // Performs one MCL iterations
    void iteration(Mat &mclmat, int e, double r, double prune_min) {
        //printf("\t\t\t## iteration ##\n");
        expand(mclmat, e);
        inflate(mclmat, r);
        prune(mclmat, prune_min);
    }

    // Performs MCL iterations until convergence is reached
    void converge(Mat &mclmat, int e, double r, double prune_min) {
        normalize(mclmat);

        // iterate until no change is found
        Mat prev; 
        int iters = 0;
        bool same = false;
        while(!same) {
            //printf("\t\t\tconverge iter %d\n", iters);
            prev = mclmat.clone();
            iters++;
            // MCL
            iteration(mclmat, e, r, prune_min);

            // Check Prev
            Mat diff;
            absdiff(mclmat, prev, diff);
            prune(diff, COMP_EPSILON);
            same = (countNonZero(diff) == 0);
        }
        prev.release();
    }

    // Markov Clustering - Runs MCL iterations on src
    void cluster(Mat &mclmat, int iters, int e, double r, double prune_min) {
        if(iters <= 0)
            converge(mclmat, e, r, prune_min);
        else {
            normalize(mclmat);
            for(int i = 0; i < iters; i++)
                iteration(mclmat, e, r, prune_min);
        }
    }

}}
