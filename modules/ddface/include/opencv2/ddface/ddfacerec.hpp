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
    
    
    // Prediction Algorithm Setters
    CV_WRAP virtual void setAlgToUse(int alg) = 0;
    CV_WRAP virtual void setLabelsToCheck(int min, double ratio) = 0;
    CV_WRAP virtual void setClustersToCheck(int min, double ratio) = 0;
    CV_WRAP virtual void setMCLSettings(int numIters, int e, double r) = 0;
    CV_WRAP virtual void setClusterSettings(double tierStep, int numTiers, int maxIters) = 0;

    // Prediction Algorithm Getters
    CV_WRAP virtual int getAlgUsed() const = 0;
    CV_WRAP virtual int getLabelsToCheckMin() const = 0;
    CV_WRAP virtual double getLabelsToCheckRatio() const = 0;
    CV_WRAP virtual int getClustersToCheckMin() const = 0;
    CV_WRAP virtual double getClustersToCheckRatio() const = 0;
    CV_WRAP virtual int getMCLIters() const = 0;
    CV_WRAP virtual int getMCLExpansionPower() const = 0;
    CV_WRAP virtual double getMCLInflationPower() const = 0;
    CV_WRAP virtual double getClusterTierStep() const = 0;
    CV_WRAP virtual int getClusterNumTiers() const = 0;
    CV_WRAP virtual int getClusterMaxIters() const = 0;

    // Broad Information Getters
    CV_WRAP virtual void getLabelInfo(OutputArray labelinfo) const = 0;
    CV_WRAP virtual int getNumLabels() const = 0;
    CV_WRAP virtual int getTotalHists() const = 0;


    // Label Specific Information Getters
    CV_WRAP virtual bool isTrainedFor(int label) const = 0;
    CV_WRAP virtual int getNumHists(int label) const = 0;
    CV_WRAP virtual int getNumClusters(int label) const = 0;


    // Threading Setters/Getters
    CV_WRAP virtual void setMaxThreads(int max) = 0;
    CV_WRAP virtual int getMaxThreads() const = 0;

    // ???: do we need this anymore?
    CV_WRAP virtual void setUseClusters(bool flag) = 0;

    CV_WRAP virtual bool load() = 0;


    // Core Prediction Algorithms
    CV_WRAP virtual void predictMulti(InputArray _src, OutputArray _preds, int numPreds) const = 0;
    CV_WRAP virtual void predictMulti(InputArray _src, OutputArray _preds, int numPreds, InputArray _labels) const = 0;
    CV_WRAP virtual void predictAll(std::vector<Mat> &_src, std::vector<Mat> &_preds, int numPreds, InputArray _labels) const = 0;

};
CV_EXPORTS_W Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX, String modelpath="");

}} 

#endif //__OPENCV_DDFACEREC_HPP__
