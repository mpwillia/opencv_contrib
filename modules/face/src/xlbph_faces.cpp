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
#include <cstring>

namespace cv { namespace face {

// Face Recognition based on Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
class xLBPH : public xLBPHFaceRecognizer
{
private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    String _modelpath;
    std::map<int, int> _labelinfo;

    // Computes a xLBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);
    
    //--------------------------------------------------------------------------
    // Additional Private Functions
    //--------------------------------------------------------------------------
    bool saveHistograms(int label, const std::vector<Mat> &histograms) const;
    bool updateHIstograms(int label, const std::vector<Mat> &histrograms) const;
    bool loadHistograms(int label, std::vector<Mat> &histograms);

    int getHistogramSize() const;
    bool matsEqual(const Mat &a, const Mat &b) const;
    
    String getHistogramsDir() const;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes this xLBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    xLBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX) :
        _grid_x(gridx),
        _grid_y(gridy),
        _radius(radius_),
        _neighbors(neighbors_),
        _threshold(threshold) {}

    // Initializes and computes this xLBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    xLBPH(InputArrayOfArrays src,
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

    ~xLBPH() { }

    // Computes a xLBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this xLBPH model with images in src and
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
    
    CV_IMPL_PROPERTY(int, GridX, _grid_x)
    CV_IMPL_PROPERTY(int, GridY, _grid_y)
    CV_IMPL_PROPERTY(int, Radius, _radius)
    CV_IMPL_PROPERTY(int, Neighbors, _neighbors)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
    
    String getModelPath() const;
    String getModelName() const;

    //--------------------------------------------------------------------------
    // Additional Public Functions 
    // NOTE: Remember to add header to opencv2/face/facerec.hpp
    //--------------------------------------------------------------------------
    void load_segmented(const String &parent_dir, const String &modelname);
    void save_segmented(const String &parent_dir, const String &modelname, bool binary_hist) const;
    bool verifyBinaryFiles(const String &parent_dir, const String &modelname);
    //void train_segmented(InputArrayOfArrays _in_src, InputArray _in_labels, const String &parent_dir, const String &modelname, bool binary_hists);
    
    void test();
};


//------------------------------------------------------------------------------
// Additional Functions and File IO
//------------------------------------------------------------------------------
void xLBPH::test() {
    _modelpath = "/images/saved-models/xLBPH-tests";
}


String xLBPH::getModelPath() const {
    return _modelpath; 
}




bool xLBPH::verifyBinaryFiles(const String &parent_dir, const String &modelname) {
    
    String modelname_bin(modelname + "-bin");
    String model_dir_bin(parent_dir + "/" + modelname_bin);
    String modelname_yaml(modelname + "-yaml");
    String model_dir_yaml(parent_dir + "/" + modelname_yaml);

    // save our model with both yaml and binary
    save_segmented(parent_dir, modelname_bin, true);
    save_segmented(parent_dir, modelname_yaml, false);

    // load info file
    String infofilepath(model_dir_bin + "/" + modelname_bin + ".yml");
    FileStorage infofile(infofilepath, FileStorage::READ);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File '" + infofilepath + "' can't be opened for writing!");
    
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
   
    for(size_t i = 0; i < labels.size(); i++) {

        char label[16];
        sprintf(label, "%d", labels.at((int)i));

        //String histfilename_bin(model_dir_bin + "/" + modelname_bin + "-histograms" + "/" + modelname_bin + "-" + label + ".bin");
        String histfilename_yaml(model_dir_yaml + "/" + modelname_yaml + "-histograms" + "/" + modelname_yaml + "-" + label + ".yml");
        
        std::vector<Mat> hists_bin;
        std::vector<Mat> hists_yaml;
        FileStorage yaml(histfilename_yaml, FileStorage::READ);
        if (yaml.isOpened()) {
            // attempt to load yaml 
            yaml["histograms"] >> hists_yaml;
        } 
        else {
            std::cout << "Cannot load YAML histograms for label " << label << " | " << histfilename_yaml << "\n";
            return false;
        }

        yaml.release();

        // attempt to load binary
        if (!loadHistograms(labels.at((int)i), hists_bin)) {
            // loading binary failed
            std::cout << "Cannot load Binary histograms for label " << label << " | " << histfilename_bin << "\n";
            return false;
        } 
      
        if(hists_yaml.size() != hists_bin.size()) {
            std::cout << "Different number of histograms between YAML(" << (int)hists_yaml.size() << ") and Binary(" << (int)hists_bin.size() << ")!\n";
            return false;
        }

        for(size_t j = 0; j < hists_yaml.size() && j < hists_bin.size(); j++) {
            if(!matsEqual(hists_yaml.at((int)j), hists_bin.at((int)j))) {
                std::cout << " -> NOT EQUAL!!! <- \n";
                return false;
            }
        }
    }

    std::cout << "Binary files are OK\n";

    return true;
} 





bool xLBPH::matsEqual(const Mat &a, const Mat &b) const {
    return countNonZero(a!=b) == 0; 
}

int xLBPH::getHistogramSize() const {
    return (int)(std::pow(2.0, static_cast<double>(_neighbors)) * _grid_x * _grid_y);
}


bool xLBPH::loadHistograms(int label, std::vector<Mat> &histograms) {

    char labelstr[16];
    sprintf(labelstr, "%d", label);
    String filename(modelpath + "/" + )
    FILE *fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    
    float buffer[getHistogramSize()];
    while(fread(buffer, sizeof(float), getHistogramSize(), fp) > 0) {
        Mat hist = Mat::zeros(1, getHistogramSize(), CV_32FC1);
        memcpy(hist.ptr<float>(), buffer, getHistogramSize() * sizeof(float));
        histograms.push_back(hist);
    }
    fclose(fp);
    return true;
}

bool xLBPH::saveHistograms(int label, const std::vector<Mat> &histograms) const {
    char labelstr[16];
    sprintf(labelstr, "%d", label);
    String filename(modelpath + "/" + )
    FILE *fp = fopen(filename.c_str(), "w");
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    
    float* buffer = new float[getHistogramSize() * (int)histograms.size()];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        memcpy((buffer + sampleIdx * getHistogramSize()), histograms.at((int)sampleIdx).ptr<float>(), getHistogramSize() * sizeof(float));
    }
    fwrite(buffer, sizeof(float), getHistogramSize() * (int)histograms.size(), fp);
    delete buffer;

    //TODO: Either increase write buffer or group all hists into one write call
    /*
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        Mat hist = histograms.at((int)sampleIdx);
        fwrite(hist.ptr<float>(), sizeof(float), getHistogramSize(), fp);
    }
    */
    fclose(fp);
    return true;
}



void xLBPH::load_segmented(const String &parent_dir, const String &modelname) {
    
    String model_dir(parent_dir + "/" + modelname);
    String infofilepath(model_dir + "/" + modelname + ".yml");
   
    FileStorage infofile(infofilepath, FileStorage::READ);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File '" + infofilepath + "' can't be opened for writing!");
    
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
        std::cout << "Loading " << (int)i << " / " << (int)labels.size() << "\r" << std::flush;
        //std::cout << "loading label '" << labels.at((int)i) << "'\r";

        char label[16];
        sprintf(label, "%d", labels.at((int)i));
        String histfilename_base(histograms_dir + "/" + modelname + "-" + label);
        String histfilename_yaml(histfilename_base + ".yml");
        //String histfilename_bin(histfilename_base + ".bin");
        
        std::vector<Mat> hists;
        FileStorage yaml(histfilename_yaml, FileStorage::READ);
        if (yaml.isOpened()) {
            // attempt to load yaml 
            yaml["histograms"] >> hists;
        } 
        // attempt to load binary
        else if (!loadHistograms(labels.at((int)i), hists)) {
            // loading binary failed
            std::cout << "cannot load histograms for label " << label << "\n";
        } 
        yaml.release();
    }
    std::cout << "Finished loading " << (int)labels.size() << " label's histograms\n";

}



void xLBPH::save_segmented(const String &parent_dir, const String &modelname, bool binary_hists) const {
   
    // create our model dir
    String model_dir(parent_dir + "/" + modelname);

    // can write WAY faster if the dir doesn't exist already
    system(("rm -rf " + model_dir).c_str());
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
    String infofilepath(model_dir + "/" + modelname + ".yml");
    FileStorage infofile(infofilepath, FileStorage::WRITE);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");

    infofile << "radius" << _radius;
    infofile << "neighbors" << _neighbors;
    infofile << "grid_x" << _grid_x;
    infofile << "grid_y" << _grid_y;
    infofile << "numlabels" << (int)histograms_map.size();
    //infofile << "labels" << unique_labels;
    infofile << "label_info" << "{";
    infofile << "labels" << unique_labels;
    infofile << "numhists" << label_num_hists;
    infofile << "}";
    infofile.release();

    // create our histogram directory
    String histogram_dir(model_dir + "/" + modelname + "-histograms");
    system(("mkdir " + histogram_dir).c_str());
    
    std::cout << "\n";
    for(size_t idx = 0; idx < unique_labels.size(); idx++) {
        std::cout << "Saving label " << (int)idx << " / " << (int)unique_labels.size() << "\r" << std::flush;
        char label[16];
        sprintf(label, "%d", unique_labels.at(idx));
        String histogram_filename(histogram_dir + "/" + modelname + "-" + label + ".yml");
        
        if(binary_hists) {
            //String histogram_rawfilename(histogram_dir + "/" + modelname + "-" + label + ".bin");
            saveHistograms(unique_labels.at(idx), histograms_map.at(unique_labels.at(idx)));
        }
        else {
            FileStorage histogram_file(histogram_filename, FileStorage::WRITE);
            if(!histogram_file.isOpened())
                CV_Error(Error::StsError, "Histogram file can't be opened for writing!");

            histogram_file << "histograms" << histograms_map.at(unique_labels.at(idx));
            histogram_file.release();
        }
    } 
    std::cout << "Finished saving " << (int)unique_labels.size() << "\n";

} 


//------------------------------------------------------------------------------
// Standard Functions and File IO
//------------------------------------------------------------------------------

// See FaceRecognizer::load.
/* TODO: Rewrite for xLBPH
 * sets modelpath
 * loads alg settings from infofile
 * loads labelinfo from infofile
 */
void xLBPH::load(const FileStorage& fs) {

}

// See FaceRecognizer::save.
/* TODO: Rewrite for xLBPH
 * wha does this do?
 * write infofile
 */
void xLBPH::save(FileStorage& fs) const {

}

void xLBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void xLBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;

    this->train(_in_src, _in_labels, true);
}


//------------------------------------------------------------------------------
// xLBPH
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

/* TODO Rewrite for xLBPH
 * sets modelpath
 * calculates histograms
 * saves histograms
 * updates lableinfo
 * saves infofile
 */
void xLBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {
    
}

/* TODO Rewrite for xLBPH
 */
void xLBPH::predict(InputArray _src, int &minClass, double &minDist) const {

}

int xLBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius, int neighbors,
                                             int grid_x, int grid_y, double threshold)
{
    return makePtr<xLBPH>(radius, neighbors, grid_x, grid_y, threshold);
}

}}