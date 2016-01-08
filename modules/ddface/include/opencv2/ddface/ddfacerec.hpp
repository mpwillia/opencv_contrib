// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DDFACEREC_HPP__
#define __OPENCV_DDFACEREC_HPP__

#include "opencv2/face.hpp"
#include "opencv2/core.hpp"

namespace cv { namespace face {

class CV_EXPORTS_W xLBPHFaceRecognizer : public FaceRecognizer 
{
public:
    /** @see setGridX */
    CV_WRAP virtual int getGridX() const = 0;
    /** @copybrief getGridX @see getGridX */
    CV_WRAP virtual void setGridX(int val) = 0;
    /** @see setGridY */
    CV_WRAP virtual int getGridY() const = 0;
    /** @copybrief getGridY @see getGridY */
    CV_WRAP virtual void setGridY(int val) = 0;
    /** @see setRadius */
    CV_WRAP virtual int getRadius() const = 0;
    /** @copybrief getRadius @see getRadius */
    CV_WRAP virtual void setRadius(int val) = 0;
    /** @see setNeighbors */
    CV_WRAP virtual int getNeighbors() const = 0;
    /** @copybrief getNeighbors @see getNeighbors */
    CV_WRAP virtual void setNeighbors(int val) = 0;
    /** @see setThreshold */
    CV_WRAP virtual double getThreshold() const = 0;
    /** @copybrief getThreshold @see getThreshold */
    CV_WRAP virtual void setThreshold(double val) = 0;

    CV_WRAP virtual void test() = 0; 

    CV_WRAP virtual void setModelPath(String modelpath) = 0;
    CV_WRAP virtual String getModelPath() const = 0; 
    CV_WRAP virtual String getModelName() const = 0;
    CV_WRAP virtual String getInfoFile() const = 0; 
    CV_WRAP virtual String getHistogramsDir() const = 0;
    CV_WRAP virtual String getHistogramFile(int label) const = 0;
    CV_WRAP virtual String getHistogramAveragesFile() const = 0;
    
    CV_WRAP virtual void setAlgToUse(int alg) = 0;
    CV_WRAP virtual void setNumThreads(int numThreads) = 0;

    CV_WRAP virtual void setMCLSettings(int numIters, int e, double r) = 0;
    CV_WRAP virtual void setClusterSettings(double tierStep, int numTiers) = 0;

    CV_WRAP virtual void setUseClusters(bool flag) = 0;

    CV_WRAP virtual void load() = 0;
    
    CV_WRAP virtual void predictMulti(InputArray _src, OutputArray _preds, int numPreds) const = 0;

};
CV_EXPORTS_W Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX, String modelpath="");

}} 

#endif //__OPENCV_DDFACEREC_HPP__
