// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_FACEREC_HPP__
#define __OPENCV_FACEREC_HPP__

#include "opencv2/face.hpp"
#include "opencv2/core.hpp"

namespace cv { namespace face {

//! @addtogroup face
//! @{

// base for two classes
class CV_EXPORTS_W BasicFaceRecognizer : public FaceRecognizer
{
public:
    /** @see setNumComponents */
    CV_WRAP virtual int getNumComponents() const = 0;
    /** @copybrief getNumComponents @see getNumComponents */
    CV_WRAP virtual void setNumComponents(int val) = 0;
    /** @see setThreshold */
    CV_WRAP virtual double getThreshold() const = 0;
    /** @copybrief getThreshold @see getThreshold */
    CV_WRAP virtual void setThreshold(double val) = 0;
    CV_WRAP virtual std::vector<cv::Mat> getProjections() const = 0;
    CV_WRAP virtual cv::Mat getLabels() const = 0;
    CV_WRAP virtual cv::Mat getEigenValues() const = 0;
    CV_WRAP virtual cv::Mat getEigenVectors() const = 0;
    CV_WRAP virtual cv::Mat getMean() const = 0;
};

/**
@param num_components The number of components (read: Eigenfaces) kept for this Principal
Component Analysis. As a hint: There's no rule how many components (read: Eigenfaces) should be
kept for good reconstruction capabilities. It is based on your input data, so experiment with the
number. Keeping 80 components should almost always be sufficient.
@param threshold The threshold applied in the prediction.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.
-   **THE EIGENFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.
-   This model does not support updating.

### Model internal data:

-   num_components see createEigenFaceRecognizer.
-   threshold see createEigenFaceRecognizer.
-   eigenvalues The eigenvalues for this Principal Component Analysis (ordered descending).
-   eigenvectors The eigenvectors for this Principal Component Analysis (ordered by their
    eigenvalue).
-   mean The sample mean calculated from the training data.
-   projections The projections of the training data.
-   labels The threshold applied in the prediction. If the distance to the nearest neighbor is
    larger than the threshold, this method returns -1.
 */
CV_EXPORTS_W Ptr<BasicFaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);

/**
@param num_components The number of components (read: Fisherfaces) kept for this Linear
Discriminant Analysis with the Fisherfaces criterion. It's useful to keep all components, that
means the number of your classes c (read: subjects, persons you want to recognize). If you leave
this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the
correct number (c-1) automatically.
@param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
is larger than the threshold, this method returns -1.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.
-   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.
-   This model does not support updating.

### Model internal data:

-   num_components see createFisherFaceRecognizer.
-   threshold see createFisherFaceRecognizer.
-   eigenvalues The eigenvalues for this Linear Discriminant Analysis (ordered descending).
-   eigenvectors The eigenvectors for this Linear Discriminant Analysis (ordered by their
    eigenvalue).
-   mean The sample mean calculated from the training data.
-   projections The projections of the training data.
-   labels The labels corresponding to the projections.
 */
CV_EXPORTS_W Ptr<BasicFaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);

class CV_EXPORTS_W LBPHFaceRecognizer : public FaceRecognizer
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
    CV_WRAP virtual std::vector<cv::Mat> getHistograms() const = 0;
    CV_WRAP virtual cv::Mat getLabels() const = 0;

    CV_WRAP virtual void load_segmented(const String &parent_dir, const String &modelname) = 0;
    CV_WRAP virtual void save_segmented(const String &parent_dir, const String &modelname, bool binary_hists) const = 0;

    CV_WRAP virtual bool verifyBinaryFiles(const String &parent_dir, const String &modelname) = 0;
   //CV_WRAP virtual void train_segmented(InputArrayOfArrays _in_src, InputArray _in_labels, const String &parent_dir, const String &modelname, bool binary_hists);

};

/**
@param radius The radius used for building the Circular Local Binary Pattern. The greater the
radius, the
@param neighbors The number of sample points to build a Circular Local Binary Pattern from. An
appropriate value is to use `8` sample points. Keep in mind: the more sample points you include,
the higher the computational cost.
@param grid_x The number of cells in the horizontal direction, 8 is a common value used in
publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
feature vector.
@param grid_y The number of cells in the vertical direction, 8 is a common value used in
publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
feature vector.
@param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
is larger than the threshold, this method returns -1.

### Notes:

-   The Circular Local Binary Patterns (used in training and prediction) expect the data given as
    grayscale images, use cvtColor to convert between the color spaces.
-   This model supports updating.

### Model internal data:

-   radius see createLBPHFaceRecognizer.
-   neighbors see createLBPHFaceRecognizer.
-   grid_x see createLBPHFaceRecognizer.
-   grid_y see createLBPHFaceRecognizer.
-   threshold see createLBPHFaceRecognizer.
-   histograms Local Binary Patterns Histograms calculated from the given training data (empty if
    none was given).
-   labels Labels corresponding to the calculated Local Binary Patterns Histograms.
 */
CV_EXPORTS_W Ptr<LBPHFaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

//! @}

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


    CV_WRAP virtual void load_segmented(const String &parent_dir, const String &modelname) = 0;
    CV_WRAP virtual void save_segmented(const String &parent_dir, const String &modelname, bool binary_hists) const = 0;

    CV_WRAP virtual bool verifyBinaryFiles(const String &parent_dir, const String &modelname) = 0;
   //CV_WRAP virtual void train_segmented(InputArrayOfArrays _in_src, InputArray _in_labels, const String &parent_dir, const String &modelname, bool binary_hists);
   
   CV_WRAP virtual void test() = 0;

};
CV_EXPORTS_W Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);


}} //namespace cv::face

#endif //__OPENCV_FACEREC_HPP__
