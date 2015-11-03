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
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#define COMP_ALG HISTCMP_CHISQR_ALT
//#define COMP_ALG HISTCMP_BHATTACHARYYA
#define SIZEOF_CV_32FC1 4

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
    // alg settings
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;
    
    // model path info
    String _modelpath;
    String _modelname;

    // label info
    std::map<int, int> _labelinfo;

    // histograms
    std::map<int, std::vector<Mat> > _histograms;
    
    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    //--------------------------------------------------------------------------
    // Additional Private Functions
    //--------------------------------------------------------------------------
    bool readHistograms(const String &filename, std::vector<Mat> &histograms) const;
    bool writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const;

    bool saveHistograms(int label, const std::vector<Mat> &histograms) const;
    bool updateHistograms(int label, const std::vector<Mat> &histrograms) const;
    bool loadHistograms(int label, std::vector<Mat> &histograms) const;
    
    bool calcHistogramAverages() const;
    bool loadHistogramAverages(std::map<int, Mat> &histavgs) const;

    bool exists(const String &filename) const;
    int getHistogramSize() const;
    bool matsEqual(const Mat &a, const Mat &b) const;

    void mmapHistograms();
    void munmapHistograms();

public:
    using FaceRecognizer::save;
    //using FaceRecognizer::load;

    // Initializes this xLBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    xLBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            String modelpath="") :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold) {
        setModelPath(modelpath);
    }

    // Initializes and computes this xLBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    xLBPH(InputArrayOfArrays src,
            InputArray labels,
            int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            String modelpath="") :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold) {
        setModelPath(modelpath);
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
    void load(const String &filename);
    void load();

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;
    
    CV_IMPL_PROPERTY(int, GridX, _grid_x)
    CV_IMPL_PROPERTY(int, GridY, _grid_y)
    CV_IMPL_PROPERTY(int, Radius, _radius)
    CV_IMPL_PROPERTY(int, Neighbors, _neighbors)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
    
    // path getters/setters
    void setModelPath(String modelpath);
    String getModelPath() const;
    String getModelName() const;
    String getInfoFile() const;
    String getHistogramsDir() const;
    String getHistogramFile(int label) const;
    String getHistogramAveragesFile() const;

    //--------------------------------------------------------------------------
    // Additional Public Functions 
    // NOTE: Remember to add header to opencv2/face/facerec.hpp
    //--------------------------------------------------------------------------
    
    void test();
};

//------------------------------------------------------------------------------
// Model Path and Model File Getters/Setters 
//------------------------------------------------------------------------------

// Sets _modelpath, extracts model name from path, and sets _modelname

void xLBPH::setModelPath(String modelpath) {
    
    // given path can't be empty
    CV_Assert(modelpath.length() > 0);

    // path can't contain "//"
    CV_Assert((int)modelpath.find("//") == -1);
    
    // find last index of '/' 
    size_t idx = modelpath.find_last_of('/');

    if((int)idx < 0) {
        _modelpath = modelpath;
        _modelname = modelpath;
    }
    else if((int)idx >= (int)modelpath.length()-1) {
        setModelPath(modelpath.substr(0, modelpath.length()-1));
    }
    else {
        _modelpath = modelpath;
        _modelname = _modelpath.substr(idx + 1);
    }

}

String xLBPH::getModelPath() const {
    return _modelpath; 
}

String xLBPH::getModelName() const {
    return _modelname;
} 

String xLBPH::getInfoFile() const {
    return getModelPath() + "/" + getModelName() + ".yml";
}

String xLBPH::getHistogramsDir() const {
    return getModelPath() + "/" + getModelName() + "-histograms";
}

String xLBPH::getHistogramFile(int label) const {
    char labelstr[16];
    sprintf(labelstr, "%d", label);
    return getHistogramsDir() + "/" + getModelName() + "-" + labelstr + ".bin";
}

String xLBPH::getHistogramAveragesFile() const {
    return getHistogramsDir() + "/" + getModelName() + "-averages.bin";
}

//------------------------------------------------------------------------------
// Additional Functions and File IO
//------------------------------------------------------------------------------
static String matToHex(const Mat &mat) {
    String s = "";
    
    const unsigned char* data = mat.ptr<unsigned char>();

    for(int i = 0; i < mat.cols; i++) {
        char valuestr[32];
        int idx = i*4;
        sprintf(valuestr, "%02x%02x ", *(data+idx), *(data+idx+1));
        s += valuestr;
        sprintf(valuestr, "%02x%02x ", *(data+idx+2), *(data+idx+3));
        s += valuestr;
    }
    return s;
}

static String matToString(const Mat &mat) {
    String s = "[";

    for(int i = 0; i < mat.cols; i++) {
        if(i != 0)
            s += ", ";

        char valuestr[64];
        sprintf(valuestr, "%f", mat.at<float>(i));
        s += valuestr;
    }
    s += "]";
    
    return s;
}

void xLBPH::test() {
    // make some fake hists
    int numhists = 16;
    int size = 4;
    std::vector<Mat> histsToSave;
    
    std::cout << "Making test hists...\n";
    for(int i = 0; i < numhists - 2; i++) {
        Mat mat = Mat::zeros(1, size, CV_32FC1);
        mat += i;
        histsToSave.push_back(mat);
    }
    Mat matmax = Mat::zeros(1, size, CV_32FC1);
    Mat matmin = Mat::zeros(1, size, CV_32FC1);
    matmax = FLT_MAX;
    matmin = FLT_MIN;
    histsToSave.push_back(matmax);
    histsToSave.push_back(matmin);
    
    std::cout << "Saving test hists...\n";
    String testhistsfile = getHistogramsDir() + "/testhists.bin";
    // write them to a fake file
    FILE *writefp = fopen(testhistsfile.c_str(), "w");
    if(writefp == NULL) 
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");

    float* writebuffer = new float[size * numhists];
    for(size_t sampleIdx = 0; (int)sampleIdx < numhists; sampleIdx++) {
        memcpy((writebuffer + sampleIdx * size), histsToSave.at((int)sampleIdx).ptr<float>(), size * sizeof(float));
    }
    fwrite(writebuffer, sizeof(float), size * numhists, writefp);
    delete writebuffer;
    fclose(writefp);
  

    std::cout << "Mapping test hists file...\n";
    // mmap fake file
    int fd = open(testhistsfile.c_str(), O_RDONLY);
    if(fd < 0)
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");

    char* mapPtr = (char*)mmap(NULL, size * numhists * sizeof(float), PROT_READ, MAP_PRIVATE, fd, 0);
    if(mapPtr == MAP_FAILED)
        CV_Error(Error::StsError, "Cannot mem map file '"+testhistsfile+"'");
    
    std::cout << "Loading query mats...\n";
    // make matricies from map
    std::vector<Mat> query;
    for(int i = 0; i < numhists; i++) {
        Mat mat(1, size, CV_32FC1, mapPtr + (size * sizeof(float) * i));
        query.push_back(mat);
    }

    
    std::cout << "Loading check mat...\n";
    // load mats from file 
    std::vector<Mat> check;
    FILE *readfp = fopen(testhistsfile.c_str(), "r");
    if(readfp == NULL) 
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");
    
    float readbuffer[size];
    while(fread(readbuffer, sizeof(float), size, readfp) > 0) {
        Mat hist = Mat::zeros(1, size, CV_32FC1);
        memcpy(hist.ptr<float>(), readbuffer, size * sizeof(float));
        check.push_back(hist);
    }
    fclose(readfp);

    std::cout << "Comparing results...\n";
    // compare results
    CV_Assert(query.size() == check.size());
    CV_Assert(query.size() == histsToSave.size());

    std::cout << "saved size: " << histsToSave.size() << "  |  query size: " << query.size() << "  |  check size: " << check.size() << "\n";
    for(size_t idx = 0; idx < query.size(); idx++) {
        
        std::cout << "idx: " << idx << std::flush;
        std::cout << "  |  saved: " << matToHex(histsToSave.at(idx)) << std::flush;
        std::cout << "  |  query: " << matToHex(query.at(idx)) << std::flush;
        std::cout << "  |  check: " << matToHex(check.at(idx)) << std::flush;
        std::cout << "\n";
        //std::cout << "saved: " << matToString(histsToSave.at(idx)) <<"  |  query: " << matToString(query.at(idx)) << "  |  check: " << matToString(check.at(idx)) << "\n";
        if(!matsEqual(query.at(idx), check.at(idx)))
        {
            //std::cout << "query: " << matToString(query.at(idx)) << "  |  " << matToString(check.at(idx)) << " :check" << "\n";
            CV_Error(Error::StsError, "MATS NOT EQUAL!!!");
        }
    }
}


bool xLBPH::matsEqual(const Mat &a, const Mat &b) const {
    return countNonZero(a!=b) == 0; 
}

int xLBPH::getHistogramSize() const {
    return (int)(std::pow(2.0, static_cast<double>(_neighbors)) * _grid_x * _grid_y);
}

bool xLBPH::exists(const String &filepath) const {
    struct stat buffer;   
    return (stat (filepath.c_str(), &buffer) == 0);   
}


// Wrapper functions for load/save/updating histograms for specific labels
bool xLBPH::loadHistograms(int label, std::vector<Mat> &histograms) const {
    return readHistograms(getHistogramFile(label), histograms);
}

bool xLBPH::saveHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getHistogramFile(label), histograms, false);
}

bool xLBPH::updateHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getHistogramFile(label), histograms, true);
}


// Main read/write functions for histograms
bool xLBPH::readHistograms(const String &filename, std::vector<Mat> &histograms) const {
    FILE *fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    //SIZEOF_CV_32FC1
    
    /*
    unsigned char buffer[getHistogramSize() * SIZEOF_CV_32FC1];
    while(fread(buffer, sizeof(unsigned char), getHistogramSize() * SIZEOF_CV_32FC1, fp) > 0) {
        Mat hist = Mat::zeros(1, getHistogramSize(), CV_32FC1);
        memcpy(hist.ptr<unsigned char>(), buffer, getHistogramSize() * SIZEOF_CV_32FC1);
        histograms.push_back(hist);
    }
    */
    
    float buffer[getHistogramSize()];
    while(fread(buffer, sizeof(float), getHistogramSize(), fp) > 0) {
        Mat hist = Mat::zeros(1, getHistogramSize(), CV_32FC1);
        memcpy(hist.ptr<float>(), buffer, getHistogramSize() * sizeof(float));
        histograms.push_back(hist);
    }
    fclose(fp);
    return true;
}


bool xLBPH::writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const {
    FILE *fp = fopen(filename.c_str(), (appendhists == true ? "a" : "w"));
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    /*   
    unsigned char* buffer = new unsigned char[getHistogramSize() * (int)histograms.size() * SIZEOF_CV_32FC1];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        memcpy(buffer + (sampleIdx * getHistogramSize() * SIZEOF_CV_32FC1), histograms.at((int)sampleIdx).ptr<unsigned char>(), getHistogramSize() * SIZEOF_CV_32FC1);
    }
    fwrite(buffer, sizeof(unsigned char), getHistogramSize() * (int)histograms.size() * SIZEOF_CV_32FC1, fp);
    delete buffer;
    */

    float* buffer = new float[getHistogramSize() * (int)histograms.size()];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        float* writeptr = buffer + ((int)sampleIdx * getHistogramSize());
        /*
        if(sampleIdx < 5)
            std::cout << "writing: " << matToHex(histograms.at(sampleIdx)) << "\n";
        */
        //printf("sampleIdx %d -> writeptr: %p\n", (int)sampleIdx, writeptr);
        memcpy(writeptr, histograms.at((int)sampleIdx).ptr<float>(), getHistogramSize() * sizeof(float));
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

bool xLBPH::calcHistogramAverages() const {
    //compareHist(histograms.at(histIdx), query, HISTCMP_CHISQR_ALT);
   
    std::vector<Mat> averages;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        std::vector<Mat> hists;
        loadHistograms(it->first, hists);
        Mat histavg = Mat::zeros(1, getHistogramSize(), CV_64FC1); // NOTE: is 64bit to prevent overflow
        
        for(size_t labelIdx = 0; labelIdx < hists.size(); labelIdx++) {
            Mat dst;
            hists.at((int)labelIdx).convertTo(dst, CV_64FC1);
            histavg += dst;
        }

        histavg /= it->second;
        histavg.convertTo(histavg, CV_32FC1);
        averages.push_back(histavg);
        
        /*
        double distAB = compareHist(hists.at(0), hists.at(1), HISTCMP_CHISQR_ALT);
        double distAEnd = compareHist(hists.at(0), hists.at(it->second - 1), HISTCMP_CHISQR_ALT);
        double distAvg = compareHist(hists.at(0), histavg, HISTCMP_CHISQR_ALT);

        std::cout << "distAB: " << distAB << " | distAEnd: " << distAEnd << " | distAvg: " << distAvg << "\n";
        */
    }
    
    return writeHistograms(getHistogramAveragesFile(), averages, false);
}

bool xLBPH::loadHistogramAverages(std::map<int, Mat> &histavgs) const {
    
    std::vector<Mat> hists;
    if(readHistograms(getHistogramAveragesFile(), hists) != true)
        return false;
    
    int histIdx = 0;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        histavgs[it->first] = hists.at(histIdx++);
    }

    return true;
}

//------------------------------------------------------------------------------
// Histogram Memory Mapping
//------------------------------------------------------------------------------
void xLBPH::mmapHistograms() {

    //_histograms = std::map<int, std::vector<Mat> >();
    _histograms.clear();
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        // map histogram
        String filename = getHistogramFile(it->first);
        int fd = open(filename.c_str(), O_RDONLY);
        if(fd < 0)
            CV_Error(Error::StsError, "Cannot open histogram file '"+filename+"'");

        //struct stat st;
        //stat(filename.c_str(), &st);
        //char* mapPtr = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        unsigned char* mapPtr = (unsigned char*)mmap(NULL, getHistogramSize() * it->second * SIZEOF_CV_32FC1, PROT_READ, MAP_PRIVATE, fd, 0);
        if(mapPtr == MAP_FAILED)
            CV_Error(Error::StsError, "Cannot mem map file '"+filename+"'");

        // make matricies
        for(int i = 0; i < it->second; i++) {
            Mat mat(1, getHistogramSize(), CV_32FC1, mapPtr + (getHistogramSize() * SIZEOF_CV_32FC1 * i));
            _histograms[it->first].push_back(mat);
        }

    }


    // verify our mmap'd histograms 
    for(std::map<int, std::vector<Mat> >::const_iterator it = _histograms.begin(); it != _histograms.end(); ++it) {
        
        std::vector<Mat> query = it->second;
        std::vector<Mat> check;
        loadHistograms(it->first, check);
        
        if(query.size() != check.size()) {
            std::cout << "query.size() = " << query.size() << "  |  check.size() = " << check.size() << "\n";
            CV_Error(Error::StsError, "query.size() != check.size()");
        }

        //CV_Assert(query.size() == check.size());

        for(size_t idx = 0; idx < query.size(); idx++) {
            if(!matsEqual(query.at(idx), check.at(idx)))
                CV_Error(Error::StsError, "MATS NOT EQUAL!!!");
        }
    }
    
    /*
    std::cout << "_histograms size: " << _histograms.size() << "\n";
    for(std::map<int, std::vector<Mat> >::const_iterator it = _histograms.begin(); it != _histograms.end(); ++it) {
        std::cout << it->first << " -> numhists: " << (it->second).size() << "\n";
    }
    */
    //std::cout << "_histograms size: " << _histograms.size() << "\n";
}

void xLBPH::munmapHistograms() {
     
}



//------------------------------------------------------------------------------
// Standard Functions and File IO
//------------------------------------------------------------------------------
void xLBPH::load() {
     
    // load data from the info file
    FileStorage infofile(getInfoFile(), FileStorage::READ);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File '"+getInfoFile()+"' can't be opened for reading!");
    
    // alg settings
    infofile["radius"] >> _radius;
    infofile["neighbors"] >> _neighbors;
    infofile["grid_x"] >> _grid_x;
    infofile["grid_y"] >> _grid_y;
    
    // label_info
    std::vector<int> labels;
    std::vector<int> numhists;
    FileNode label_info = infofile["label_info"];
    label_info["labels"] >> labels;
    label_info["numhists"] >> numhists;
  
    CV_Assert(labels.size() == numhists.size());
    _labelinfo = std::map<int, int>(); 
    for(size_t idx = 0; idx < labels.size(); idx++) {
        _labelinfo[labels.at((int)idx)] = numhists.at((int)idx);
    }
    infofile.release();

    // mem map histograms
    mmapHistograms();
}


void xLBPH::load(const String &modelpath) {
    
    // set our model path to the filename
    setModelPath(modelpath);

    // load the model
    load();    
}


// See FaceRecognizer::load.
/* TODO: Rewrite for xLBPH
 * sets modelpath
 * loads alg settings from infofile
 * loads labelinfo from infofile
 */
void xLBPH::load(const FileStorage& fs) {
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
}

// See FaceRecognizer::save.
/* TODO: Rewrite for xLBPH
 * wha does this do?
 * write infofile
 */
void xLBPH::save(FileStorage& fs) const {
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for reading!");
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
        String error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    
    // Get all of the labels
    std::vector<int> allLabels;
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
       allLabels.push_back(labels.at<int>((int)labelIdx));
    }
     
    std::cout << "Organizing images by label...\n";
    // organize the mats and labels
    std::map<int, std::vector<Mat> > labelImages;
    for(size_t matIdx = 0; matIdx < src.size(); matIdx++) {
        labelImages[allLabels.at((int)matIdx)].push_back(src.at((int)matIdx));
    }
    std::cout << "Organized images for " << labelImages.size() << " labels.\n";
   
    if(!preserveData)
    {
        // create model directory
        // check if the model directory already exists
        if(exists(getModelPath())) {
            // model directory exists
            // check if the directory is actually a model
            if(exists(getInfoFile()) && exists(getHistogramsDir())) {
                // is a model dir so overwrite  
                system(("rm -r " + getModelPath()).c_str());     
            }
            else {
                // is not a model dir so error
                CV_Error(Error::StsError, "Given model path at '" + getModelPath() +"' already exists and doesn't look like an xLBPH model directory; refusing to overwrite for data safety.");
            }
        }
        
        // create the model directories
        system(("mkdir " + getModelPath()).c_str()); 
        system(("mkdir " + getHistogramsDir()).c_str());
    }
    
    std::vector<int> uniqueLabels;
    std::vector<int> numhists;

    // start training
    int labelcount = 1;
    for(std::map<int, std::vector<Mat> >::const_iterator it = labelImages.begin(); it != labelImages.end(); ++it) {
        std::cout << "Calculating histograms for label " << labelcount << " / " << labelImages.size() << " [" << it->first << "]\r" << std::flush;

        //label = it->first;
        std::vector<Mat> imgs = it->second;
        std::vector<Mat> hists;

        for(size_t sampleIdx = 0; sampleIdx < imgs.size(); sampleIdx++) {
            // calculate lbp image
            Mat lbp_image = elbp(imgs.at(sampleIdx), _radius, _neighbors);

            // get spatial histogram from this lbp image
            Mat p = spatial_histogram(
                    lbp_image, /* lbp_image */
                    static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
                    _grid_x, /* grid size x */
                    _grid_y, /* grid size y */
                    true);

            hists.push_back(p);
        }
        
        uniqueLabels.push_back(it->first);
        numhists.push_back((int)imgs.size());
        writeHistograms(getHistogramFile(it->first), hists, preserveData);
        hists.clear();

        labelcount++;
        // free memory
        /*
        for (std::vector<Mat>::iterator it = hists.begin() ; it != hists.end(); ++it) {
             
        }
        */
    }
    std::cout << "Finished calculating histograms for " << labelImages.size() << " labels.            \n";
    std::cout << "Writing infofile\n";
    String infofilepath(_modelpath + "/" + getModelName() + ".yml");
    FileStorage infofile(infofilepath, FileStorage::WRITE);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");

    infofile << "radius" << _radius;
    infofile << "neighbors" << _neighbors;
    infofile << "grid_x" << _grid_x;
    infofile << "grid_y" << _grid_y;
    infofile << "numlabels" << (int)labelImages.size();
    //infofile << "labels" << unique_labels;
    infofile << "label_info" << "{";
    infofile << "labels" << uniqueLabels;
    infofile << "numhists" << numhists;
    infofile << "}";
    infofile.release();

    // lastly we need to set _labelinfo
    _labelinfo = std::map<int, int>(); // if _labelinfo was set then clear it
    for(size_t labelIdx = 0; labelIdx < uniqueLabels.size(); labelIdx++) {
        _labelinfo[uniqueLabels.at((int)labelIdx)] = numhists.at((int)labelIdx);
    }
    
    /*
    std::cout << "Calculating histogram averages...\n";
    calcHistogramAverages();
    */

    load();
    std::cout << "Histogram Size: " << getHistogramSize() << "\n";

    std::cout << "Check against standard LBPH histograms...\n";
    Ptr<LBPHFaceRecognizer> check = createLBPHFaceRecognizer(_radius, _neighbors, _grid_x, _grid_y, _threshold);
    check->train(_in_src, _in_labels);

    std::vector<cv::Mat> checkHistsAll = check->getHistograms();
    cv::Mat checkLabels = check->getLabels();
    //std::map<int, size_t> checkIdxs;
    bool allEqual = true;
    
    //sort LBPH histograms
    std::map<int, std::vector<Mat> > checkHistsMap;
    for(size_t idx = 0; idx < checkHistsAll.size(); idx++) {
        checkHistsMap[checkLabels.at<int>((int)idx)].push_back(checkHistsAll.at(idx));
    }
    
    std::cout << "#### Checking histograms loaded from file into mem...\n";
    for(std::map<int, std::vector<Mat> >::const_iterator it = checkHistsMap.begin(); it != checkHistsMap.end(); ++it) {
        std::vector<Mat> checkHists = it->second;
        //std::vector<Mat> queryHists = _histograms[it->first];
        std::vector<Mat> queryHists;
        loadHistograms(it->first, queryHists);
        
        if(checkHists.size() != queryHists.size()) {
            std::cout << "ERROR: For label " << it->first << " checkHists is not the same siee as queryHists\n";
            allEqual = false;
        }
        else
        {
            for(size_t histIdx = 0; histIdx < checkHists.size(); histIdx++) {
                if(!matsEqual(checkHists.at(histIdx), queryHists.at(histIdx))) {
                    std::cout << "ERROR: For label " << it->first << " at histIdx of " << histIdx << " hists NOT EQUAL!!!\n";
                    std::cout << "Expected: " << matToHex(checkHists.at(histIdx)) << "\n";
                    std::cout << "But Got:  " << matToHex(queryHists.at(histIdx)) << "\n";
                    allEqual = false;
                    break;
                } 
                else {
                    /*
                    std::cout << "GOOD: For label " << it->first << "\n";
                    std::cout << "Expected: " << matToHex(checkHists.at(histIdx)) << "\n";
                    std::cout << "And Got:  " << matToHex(queryHists.at(histIdx)) << "\n";
                    */
                }

            }
        }
    }
    CV_Assert(allEqual == true);


    std::cout << "\n#### Checking histograms memmapped from file...\n";
    for(std::map<int, std::vector<Mat> >::const_iterator it = checkHistsMap.begin(); it != checkHistsMap.end(); ++it) {
        std::vector<Mat> checkHists = it->second;
        std::vector<Mat> queryHists = _histograms[it->first];
        //std::vector<Mat> queryHists;
        //loadHistograms(it->first, queryHists);
        
        if(checkHists.size() != queryHists.size()) {
            std::cout << "ERROR: For label " << it->first << " checkHists is not the same siee as queryHists\n";
            allEqual = false;
        }
        else
        {
            for(size_t histIdx = 0; histIdx < checkHists.size(); histIdx++) {
                if(!matsEqual(checkHists.at(histIdx), queryHists.at(histIdx))) {
                    std::cout << "ERROR: For label " << it->first << " at histIdx of " << histIdx << " hists NOT EQUAL!!!\n";
                    std::cout << "Expected: " << matToHex(checkHists.at(histIdx)) << "\n";
                    std::cout << "But Got:  " << matToHex(queryHists.at(histIdx)) << "\n";
                    allEqual = false;
                    break;
                } 
            }
        }
    }
    CV_Assert(allEqual == true);

    /*
    for(size_t idx = 0; idx < checkHists.size(); idx++) {
        int label = checkLabels.at<int>((int)idx);

        if(!matsEqual(_histograms[label].at(checkIdxs[label]++), checkHists.at(idx)))
        {
            //std::cout << "query: " << matToString(query.at(idx)) << "  |  " << matToString(check.at(idx)) << " :check" << "\n";
            CV_Error(Error::StsError, "MATS NOT EQUAL!!!");
        }
    }
    */    



    std::cout << "Training complete\n";
}

/* TODO Rewrite for xLBPH
 */
void xLBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    
    CV_Assert((int)_labelinfo.size() > 0);
    CV_Assert((int)_histograms.size() > 0);

    /*
    if((int)_labelinfo.size() <= 0) {
        CV_Error(Error::StsError, "Given model path at '" + getModelPath() +"' already exists and doesn't look like an xLBPH model directory; refusing to overwrite for data safety.");
    }
    */

    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);

    // find labels to check
    /*
    std::map<int, Mat> histavgs;
    std::vector<std::pair<double, int> > avgsdists;
    loadHistogramAverages(histavgs);
    for(std::map<int, Mat>::const_iterator it = histavgs.begin(); it != histavgs.end(); ++it) {
        avgsdists.push_back(std::pair<double, int>(compareHist(it->second, query, COMP_ALG), it->first));
    }
    std::sort(avgsdists.begin(), avgsdists.end());

    for(size_t avgsdistsIdx = 0; avgsdistsIdx < avgsdists.size(); avgsdistsIdx++) {
        std::cout << avgsdists.at(avgsdistsIdx).first << " | " << avgsdists.at(avgsdistsIdx).second << "\n";
    }
    */

    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    
    //NOTE: <double, int> instead of <int, double> so we sort by dist not label
    std::vector<std::pair<double, int> > preds;
    std::map<int, double> bestpreds;
    std::map<int, std::vector<double> > alldists;

    //int labelcount = 0;
    for(std::map<int, std::vector<Mat> >::const_iterator it = _histograms.begin(); it != _histograms.end(); ++it) {
        
        bestpreds[it->first] = DBL_MAX;
        std::vector<Mat> hists = it->second;

        for(size_t histIdx = 0; histIdx < hists.size(); histIdx++) {
            double dist = compareHist(hists.at(histIdx), query, COMP_ALG);
            alldists[it->first].push_back(dist);
            if((dist < minDist) && (dist < _threshold)) {
                minDist = dist;
                minClass = it->first;
                preds.push_back(std::pair<double, int>(dist, it->first));

                //std::cout << "[" << minClass << " | " << minDist << "], ";
            }

            if(dist < bestpreds[it->first]) {
                bestpreds[it->first] = dist;
            }

        }

        std::sort(alldists[it->first].begin(), alldists[it->first].end());
    }
    std::sort(preds.begin(), preds.end());
    
    std::cout << "\nPredictions: ";
    for(size_t idx = 0; idx < preds.size() ; idx++) {
        std::pair<double, int> pred = preds.at(idx);
        printf("[%d, %f], ", pred.second, pred.first);
        //std::cout << pred->second << ", " << pred->first
        //std::cout << preds.at(idx)  << ", ";
    }
    std::cout << "\n";
    
    std::cout << "\nBest Prediction by PID: ";
    for(std::map<int, double>::const_iterator it = bestpreds.begin(); it != bestpreds.end(); ++it) {
        printf("[%d, %f]\n", it->first, it->second);
    }

    std::cout << "\nAll Predictions by PID: ";
    for(std::map<int, std::vector<double> >::const_iterator it = alldists.begin(); it != alldists.end(); ++it) {
        std::vector<double> dists = it->second;
        std::cout << it->first << "[" << dists.size() << " total] -> ";
        for(size_t distIdx = 0; distIdx < dists.size() && (int)distIdx < 10; distIdx++) {
            std::cout << dists.at(distIdx) << ", ";
        }
        std::cout << "\n";
    }

    /*
    // iterate through _labelinfo
    int labelcount = 0;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        //std::cout << "Calculating histogram distance for label " << labelcount << " / " << _labelinfo.size() << " [" << it->first << "]\r" << std::flush;

        std::vector<Mat> histograms;
        loadHistograms(it->first, histograms);

        CV_Assert((int)histograms.size() == it->second);

        if((int)histograms.size() > 0) {
            double avgDist = 0;
            for(size_t histIdx = 0; histIdx < histograms.size(); histIdx++) {
                avgDist += compareHist(histograms.at(histIdx), query, COMP_ALG);

                //check if it is even possible for us to be better
                if(avgDist / it->second > minDist) {
                    // if it's not then stop comparing histograms, set invalid avg dist and break
                    //std::cout << "\nTerminated calcuation early\n";
                    avgDist = -1;
                    break;
                }
            }

            if(avgDist >= 0)
            {
                avgDist /= it->second;
                if(avgDist < minDist) {
                    minDist = avgDist;
                    minClass = it->first;
                }
            }
            
            for(size_t histIdx = 0; histIdx < histograms.size(); histIdx++) {
                double dist = compareHist(histograms.at(histIdx), query, HISTCMP_CHISQR_ALT);
                if((dist < minDist) && (dist < _threshold)) {
                    minDist = dist;
                    minClass = it->first;
                }

            }
        }

        labelcount++;
    }
    */
    //std::cout << "Finished calculating histogram distance for  " << _labelinfo.size() << " labels.            \n";
}

int xLBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius, int neighbors,
                                                   int grid_x, int grid_y, 
                                                   double threshold, 
                                                   String modelpath)
{
    return makePtr<xLBPH>(radius, neighbors, grid_x, grid_y, threshold, modelpath);
}

}}
