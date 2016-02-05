
#ifndef __MODELSTORAGE_HPP 
#define __MODELSTORAGE_HPP

#include "precomp.hpp"

namespace cv { namespace face {

class ModelStorage {
private: 

   String _modelpath;
   String _modelname;
  
   // Basic Collection of Generic File/Directory Functions
   bool isDirectory(const String &filepath) const;
   bool isRegularFile(const String &filepath) const;
   bool fileExists(const String &filepath) const;
   String getFileName(const String &filepath) const;
   String getFileParent(const String &filepath) const;
   std::vector<String> listdir(const String &dirpath) const;
   bool mkdirs(const String &dirpath) const;
   bool rmr(const String &dirpath) const;

   // Model Creation/Manipulation
   void setModelPath(String path);
   bool checkModel(const String &name, const String &path) const;

public:

   ModelStorage(String path) {
      setModelPath(path);
   };

   void test() const;

   // Model Creation/Manipulation
   bool create(bool overwrite) const;

   // Model Information
   bool isValidModel() const;
   bool exists() const;
   String getPath() const;
   String getName() const;

   // Model File Getters   
   String getInfoFile() const;
   String getHistogramsDir() const;
   String getHistogramFile(int label) const;
   String getHistogramAveragesFile() const;

   // Reading/Writing Histograms
   bool loadHistograms(int label, std::vector<Mat> &histograms, int histSize) const;
   bool saveHistograms(int label, const std::vector<Mat> &histograms, int histSize) const;
   bool updateHistograms(int label, const std::vector<Mat> &histograms, int histSize) const;
   bool readHistograms(const String &filename, std::vector<Mat> &histograms, int histSize) const;
   bool writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists, int histSize) const;

};

}}

#endif

