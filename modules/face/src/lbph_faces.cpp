/*
 * Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include "precomp.hpp"
#include "opencv2/face.hpp"
#include "face_basic.hpp"

#include <cstdio>

namespace cv { namespace face {

// Face Recognition based on Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
class LBPH : public LBPHFaceRecognizer
{
private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    std::vector<Mat> _histograms;
    Mat _labels;

    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);
    
    void saveRawHistograms(const String &filename, const std::vector<Mat> &histograms) const;
    void loadRawHistograms(const String &filename, std::vector<Mat> &histograms);

    int getHistogramSize() const;
    bool matsEqual(const Mat &a, const Mat &b) const;


public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes this LBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    LBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX) :
        _grid_x(gridx),
        _grid_y(gridy),
        _radius(radius_),
        _neighbors(neighbors_),
        _threshold(threshold) {}

    // Initializes and computes this LBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    LBPH(InputArrayOfArrays src,
            InputArray labels,
            int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX) :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold) {
        train(src, labels);
    }

    ~LBPH() { }

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this LBPH model with images in src and
    // corresponding labels in labels.
    void update(InputArrayOfArrays src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;
    
    void loadTest(const String &parent_dir, const String &modelname);
    void saveTest(const String &parent_dir, const String &modelname) const;

    CV_IMPL_PROPERTY(int, GridX, _grid_x)
    CV_IMPL_PROPERTY(int, GridY, _grid_y)
    CV_IMPL_PROPERTY(int, Radius, _radius)
    CV_IMPL_PROPERTY(int, Neighbors, _neighbors)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
    CV_IMPL_PROPERTY_RO(std::vector<cv::Mat>, Histograms, _histograms)
    CV_IMPL_PROPERTY_RO(cv::Mat, Labels, _labels)
};

bool LBPH::matsEqual(const Mat &a, const Mat &b) const {
    return countNonZero(a!=b) == 0; 
}

int LBPH::getHistogramSize() const {
    return (int)(std::pow(2.0, static_cast<double>(_neighbors)) * _grid_x * _grid_y);
}

void LBPH::loadRawHistograms(const String &filename, std::vector<Mat> &histograms) {
    FILE *fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        std::cout << "cannot open file at '" << filename << "'\n";
        return;
    }
    
    float buffer[getHistogramSize()];
    while(fread(buffer, sizeof(float), getHistogramSize(), fp) > 0) {
        Mat hist = Mat::zeros(1, getHistogramSize(), CV_32FC1);
        
        for(int i = 0; i < getHistogramSize(); i++) {
            hist.at<float>(0, i) = buffer[i]; 
        }
        histograms.push_back(hist);
    }
    fclose(fp);
}

void LBPH::loadTest(const String &parent_dir, const String &modelname) {
    
    String model_dir(parent_dir + "/" + modelname);
    String filename(model_dir + "/" + modelname + ".yml");
   
   FileStorage infofile(filename, FileStorage::READ);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File '" + filename + "' can't be opened for writing!");
    
    infofile["radius"] >> _radius;
    infofile["neighbors"] >> _neighbors;
    infofile["grid_x"] >> _grid_x;
    infofile["grid_y"] >> _grid_y;

    std::vector<int> labels;
    std::vector<int> numhists;
    FileNode label_info = infofile["label_info"];
    label_info["labels"] >> labels;
    label_info["numhists"] >> numhists;
    
    infofile.release();

    std::cout << "labels: [ ";
    for(size_t i = 0; i < labels.size(); i++) {
        if(i != 0)
            std::cout << ", ";
        std::cout << labels.at((int)i);
    }
    std::cout << " ]\n";
    std::cout << "numhists: [ ";
    for(size_t i = 0; i < numhists.size(); i++) {
        if(i != 0)
            std::cout << ", ";
        std::cout << numhists.at((int)i);
    }
    std::cout << " ]\n";

    String histograms_dir(model_dir + "/" + modelname + "-histograms");
    for(size_t i = 0; i < labels.size(); i++) {
        std::cout << "loading label '" << labels.at((int)i) << "'\n";

        char label[16];
        sprintf(label, "%d", labels.at((int)i));
        String histfilename_base(histograms_dir + "/" + modelname + "-" + label);
        String histfilename_yaml(histfilename_base + ".yml");
        String histfilename_bin(histfilename_base + ".bin");
        
        std::vector<Mat> yaml_hists;
        std::vector<Mat> bin_hists;
        
        std::cout << "loading yaml...\n";
        FileStorage yaml(histfilename_yaml, FileStorage::READ);
        //readFileNodeList(yaml["histograms"], yaml_hists);
        yaml["histograms"] >> yaml_hists;
        yaml.release();

        std::cout << "loading bin...\n";
        loadRawHistograms(histfilename_bin, bin_hists);
       
        std::cout << "yaml hists size: " << yaml_hists.size() << "\n";
        std::cout << "bin hists size: " << bin_hists.size() << "\n";
       
        if(matsEqual(yaml_hists.at(0), bin_hists.at(0)))
            std::cout << "FIRSTS ARE EQUAL!!!!!\n";
        else
            std::cout << "NOT EQUAL!!!!\n";

    }

}

void LBPH::load(const FileStorage& fs) {
    fs["radius"] >> _radius;
    fs["neighbors"] >> _neighbors;
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    fs["labels"] >> _labels;
    const FileNode& fn = fs["labelsInfo"];
    if (fn.type() == FileNode::SEQ)
    {
        _labelsInfo.clear();
        for (FileNodeIterator it = fn.begin(); it != fn.end();)
        {
            LabelInfo item;
            it >> item;
            _labelsInfo.insert(std::make_pair(item.label, item.value));
        }
    }
}

void LBPH::saveRawHistograms(const String &filename, const std::vector<Mat> &histograms) const {
    FILE *fp = fopen(filename.c_str(), "w");
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        Mat hist = histograms.at((int)sampleIdx);
        fwrite(hist.data, sizeof(float), getHistogramSize(), fp);
    }
    fclose(fp);
}

void LBPH::saveTest(const String &parent_dir, const String &modelname) const {
   
    // create our model dir
    String model_dir(parent_dir + "/" + modelname);
    system(("mkdir " + model_dir).c_str());

    // create a map between our labels and our histograms 
    std::map<int, std::vector<Mat> > histograms_map;
    for(size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        histograms_map[_labels.at<int>((int)sampleIdx)].push_back(_histograms[sampleIdx]);
    }
    
    //int unique_labels[(int)histograms_map.size()];
    std::vector<int> unique_labels;
    std::vector<int> label_num_hists;
    for(std::map<int, std::vector<Mat> >::iterator it = histograms_map.begin(); it != histograms_map.end(); ++it) {
        unique_labels.push_back(it->first);
        label_num_hists.push_back((it->second).size());
    }

    // create our main info file
    String filename(model_dir + "/" + modelname + ".yml");
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");

    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    fs << "numlabels" << (int)histograms_map.size();
    //fs << "labels" << unique_labels;
    fs << "label_info" << "{";
    fs << "labels" << unique_labels;
    fs << "numhists" << label_num_hists;
    fs << "}";
    fs << "histogram_size" << getHistogramSize();
    fs.release();

    // create our histogram directory
    String histogram_dir(model_dir + "/" + modelname + "-histograms");
    system(("mkdir " + histogram_dir).c_str());

    for(size_t idx = 0; idx < unique_labels.size(); idx++) {
        char label[16];
        sprintf(label, "%d", unique_labels.at(idx));
        String histogram_filename(histogram_dir + "/" + modelname + "-" + label + ".yml");
        
        FileStorage histogram_file(histogram_filename, FileStorage::WRITE);
        if (!histogram_file.isOpened())
            CV_Error(Error::StsError, "Histogram file can't be opened for writing!");

        histogram_file << "histograms" << histograms_map.at(unique_labels.at(idx));
        histogram_file.release();
        
        String histogram_rawfilename(histogram_dir + "/" + modelname + "-" + label + ".bin");
        saveRawHistograms(histogram_rawfilename, histograms_map.at(unique_labels.at(idx)));

    } 
} 

// See FaceRecognizer::save.
void LBPH::save(FileStorage& fs) const {
    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    fs << "labels" << _labels;
    fs << "labelsInfo" << "[";
    for (std::map<int, String>::const_iterator it = _labelsInfo.begin(); it != _labelsInfo.end(); it++)
        fs << LabelInfo(it->first, it->second);
    fs << "]";
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void LBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;

    this->train(_in_src, _in_labels, true);
}


//------------------------------------------------------------------------------
// LBPH
//------------------------------------------------------------------------------

template <typename _Tp> static
void olbp_(InputArray _src, OutputArray _dst) {
    // get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2, src.cols-2, CV_8UC1);
    Mat dst = _dst.getMat();
    // zero the result matrix
    dst.setTo(0);
    // calculate patterns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i,j-1) >= center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
    default:
        String error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        CV_Error(Error::StsNotImplemented, error_msg);
        break;
    }
}

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= (int)src.total();
    }
    return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type()) {
        case CV_8SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_8UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_16SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_16UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_32SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_32FC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        default:
            CV_Error(Error::StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}


static Mat spatial_histogram(InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/)
{
    Mat src = _src.getMat();
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if(src.empty())
        return result.reshape(1,1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);

            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);

            // free memory
            src_cell.release();
            cell_hist.release();
   
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------

static Mat elbp(InputArray src, int radius, int neighbors) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {

    if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error(Error::StsBadArg, error_message);
    }
    if(_in_src.total() == 0) {
        String error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsUnsupportedFormat, error_message);
    } else if(_in_labels.getMat().type() != CV_32SC1) {
        String error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error(Error::StsUnsupportedFormat, error_message);
    }

    // get the vector of matrices
    std::vector<Mat> src;
    _in_src.getMatVector(src);
    // get the label matrix
    Mat labels = _in_labels.getMat();
    // check if data is well- aligned
    if(labels.total() != src.size()) {
        String error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }

    // if this model should be trained without preserving old data, delete old model data
    if(!preserveData) {
        _labels.release();
        _histograms.clear();
    }

    // append labels to _labels matrix
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
        _labels.push_back(labels.at<int>((int)labelIdx));
    }

    // store the spatial histograms of the original data
    for(size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors);
         
        if(sampleIdx == 0)
            std::cout << "lbp_image size = " << lbp_image.cols << "x" << lbp_image.rows << " | depth = " << lbp_image.depth() << " | channels = " << lbp_image.channels() << "\n";
         
        // get spatial histogram from this lbp image
        Mat p = spatial_histogram(
                lbp_image, /* lbp_image */
                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true);
         
        if(sampleIdx == 0)
            std::cout << "p size = " << p.cols << "x" << p.rows << " | depth = " << p.depth() << " | channels = " << p.channels() << "\n";

        // add to templates
        _histograms.push_back(p);

        // free memory
        //lbp_image.release();
    }
   
    std::cout << "Num Histograms: " << _histograms.size() << "\n";
    std::cout << "Elems In Histograms : " << _histograms.at(0).rows << "x" << _histograms.at(0).cols << "\n";

}

void LBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    if(_histograms.empty()) {
        // throw error if no data (or simply return -1?)
        String error_message = "This LBPH model is not computed yet. Did you call the train method?";
        CV_Error(Error::StsBadArg, error_message);
    }
    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    double maxDist = 0;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        double dist = compareHist(_histograms[sampleIdx], query, HISTCMP_CHISQR_ALT);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int) sampleIdx);
        }

        if(dist > maxDist)
            maxDist = dist;
    }
    std::cout << "\n  -->  Max Dist = " << maxDist << " | Min Dist = " << minDist<< "\n";
}

int LBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

Ptr<LBPHFaceRecognizer> createLBPHFaceRecognizer(int radius, int neighbors,
                                             int grid_x, int grid_y, double threshold)
{
    return makePtr<LBPH>(radius, neighbors, grid_x, grid_y, threshold);
}

}}
