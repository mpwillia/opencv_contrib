
#include "modelstorage.hpp"

#include <cstring>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>

#define SIZEOF_CV_32FC1 4

namespace cv { namespace face {

//------------------------------------------------------------------------------
// A Basic Collection of Generic File/Directory  
//------------------------------------------------------------------------------

// Returns true if the file at the given path is a directory; false otherwise
bool ModelStorage::isDirectory(const String &filepath) const {
   struct stat buffer;
   if(stat(filepath.c_str(), &buffer)==0)
      return S_ISDIR(buffer.st_mode);
   else
      return false;
} 

// Returns true if the file at the given path is a regular file; false otherwise
bool ModelStorage::isRegularFile(const String &filepath) const {
   struct stat buffer;
   if(stat(filepath.c_str(), &buffer)==0)
      return S_ISREG(buffer.st_mode);
   else
      return false;
} 

// Returns true if the file at the given path exists; false otherwise
bool ModelStorage::fileExists(const String &filepath) const {
   struct stat buffer;     
   return (stat(filepath.c_str(), &buffer) == 0);
} 

// Gets the name of the file from the given filepath
String ModelStorage::getFileName(const String &filepath) const {
   size_t idx = filepath.find_last_of('/');

   if((int)idx < 0) {
      // if we don't find a '/' at all then just return the path
      return filepath;
   }
   else if((int)idx >= (int)filepath.length()-1) {
      // if we have a trailing '/' then remove it and try again
      return getFileName(filepath.substr(0, filepath.length()-1));
   }
   return filepath.substr(idx + 1);
} 

// Gets the path of the given file's parent
String ModelStorage::getFileParent(const String &filepath) const {
    
   size_t idx = filepath.find_last_of('/');

   if((int)idx < 0) {
      // if we don't find a '/' at all then we don't know the parent 
      return "";
   }
   else if((int)idx == 0) {
      // if the last occurrence of '/' is at the start than just return root
      return "/";
   } 
   else if((int)idx >= (int)filepath.length()-1) {
      // if we have a trailing '/' then remove it and try again
      return getFileName(filepath.substr(0, filepath.length()-1));
   }
   else {
      // otherwise, strip everything after the '/'
      return filepath.substr(0, idx);
   }
} 


// Returns filepaths to all of the files within the given directory
// Assumes the given path is a valid existing directory with permissions
// If the given path isn't a directory or it doesn't exist a empty vectory will be returned
std::vector<String> ModelStorage::listdir(const String &dirpath) const{
   std::vector<String> contents;
   if(!isDirectory(dirpath))
      return contents;

   struct dirent **namelist;
   int n;

   bool check = true;
   n = scandir(dirpath.c_str(), &namelist, NULL, alphasort);
   if (n < 0)
      CV_Error(Error::StsError, "Error reading directory at '"+dirpath+"'"); 
   else {
      while (n--) {
         String name(namelist[n]->d_name);

         if(name != "." && name != "..")
            contents.push_back(dirpath + "/" + name);
         free(namelist[n]);
      }
      free(namelist);
   }
   return contents;
} 

// Makes the directory at the given filepath along with all parents if they don't exist
bool ModelStorage::mkdirs(const String &dirpath) const {

   if(fileExists(dirpath)) {
      // a file already exists at that path, can't create
      return false;
   }

   String parent = getFileParent(dirpath);
   if((int)parent.length() > 0 && !fileExists(parent)) {
      // our parent doesn't exist, so lets try to make it
      if(!mkdirs(parent)) {
         // if we failed to make our parent, return false
         return false;
      } 
   } 

   if((int)parent.length() <= 0 || isDirectory(parent)) {
      // good, our parent is a directory, so lets try to make ourselves
      // return true if we get no errors making it; false otherwise
      // dir has rwxrwxrwx perms
      /* Perms are as follows:
       * S_I<Perm><Who>
       * Where <Perm> is:        And <Who> is
       * R - Read                USR - User
       * W - Write               GRP - Group
       * X - Execute             OTH - Other
       * Shortcuts for RWX
       * S_IRWXU - RWX User
       * S_IRWXG - RWX Group
       * S_IRWXO - RWX Other
       */
      return (mkdir(dirpath.c_str(), ACCESSPERMS) == 0);
   } 
   else {
      // our parent isn't a directory, we can't make a subdirectory under a normal file
      return false; 
   } 
} 

// Removes the given file and all it's contents recursively
bool ModelStorage::rmr(const String &filepath) const {

   if(filepath == "/" || (int)filepath.length() <= 0) {
      return false; 
   } 

   if(!fileExists(filepath)) {
      // file doesn't exist, can't remove! 
      return false;
   } 
   
   // if the file is not a directory contents will be empty
   std::vector<String> contents = listdir(filepath);
   if(contents.size() > 0) {
      // remove each child, if one fails the whole thing fails
      for(String file : contents) {
         if(!rmr(file)) {
            return false;
         }
      } 
   } 
   return (remove(filepath.c_str()) == 0);
} 





//------------------------------------------------------------------------------
// Model Creation/Manipulation Functions 
//------------------------------------------------------------------------------
void ModelStorage::setModelPath(String path) {
   // given path can't be empty
   CV_Assert(path.length() > 0);
          
   // path can't contain "//"
   CV_Assert((int)path.find("//") == -1);

   // find last index of '/' 
   size_t idx = path.find_last_of('/');

   if((int)idx < 0) {
     _modelpath = path;
   }
   else if((int)idx >= (int)path.length()-1) {
     setModelPath(path.substr(0, path.length()-1));
   }
   else {
     _modelpath = path;
   }

   _modelname = getFileName(_modelpath);
}

void ModelStorage::setAlgSettings(int radius, int neighbors, int grid_x, int grid_y) {
   _alg.radius = radius;
   _alg.neighbors = neighbors;
   _alg.grid_x = grid_x;
   _alg.grid_y = grid_y;
} 

void ModelStorage::setAlgSettings(AlgSettings alg) {
   _alg.radius = alg.radius;
   _alg.neighbors = alg.neighbors;
   _alg.grid_x = alg.grid_x;
   _alg.grid_y = alg.grid_y;;
} 

bool ModelStorage::create(bool overwrite) const { 
   // check if a model already exists at our _modelpath
   if(exists()) {
      if(!overwrite) {
         // don't overwrite, told not to
         CV_Error(Error::StsError, "Cannot create model at '"+getPath()+"' - a model already exists at that path! Set <overwrite> to true to overwrite the existing model.");
         return false;
      }
      else if (!isValidModel()){
         // can't overwrite, directory at our _modelpath isn't an xLBPH model, unsafe to overwrite it 
         CV_Error(Error::StsError, "Given model path at '"+getPath()+"' already exists and doesn't look like an xLBPH model directory; refusing to overwrite for data safety.");
         return false;
      } 
      else {
         // overwrite the model by deleting the existing one
         if(!rmr(getPath())) {
            CV_Error(Error::StsError, "Given model path at '"+getPath()+"' cannot be overwritten; failed to remove the old model.");
            return false;
         } 
      }
   }

   // Start making the model
   if(!mkdirs(getPath())) {
      CV_Error(Error::StsError, "Failed to create model at '"+getPath()+"'; unable to create directory");
      return false;
   } 

   if(!mkdirs(getLabelsDir())) {
      CV_Error(Error::StsError, "Failed to create model at '"+getPath()+"'; unable to create directory");
      return false;
   } 

   return true;
} 

bool ModelStorage::writeMetadata(AlgSettings alg, std::vector<int> &labels, std::vector<int> &numhists) const {

   CV_Assert((int)labels.size() == (int)numhists.size());

   FileStorage metadata(getMetadataFile(), FileStorage::WRITE);
   if(!metadata.isOpened()) {
      CV_Error(Error::StsError, "File '"+getMetadataFile()+"' can't be opened for writing!");
      return false;
   }

   // alg settings
   metadata << "radius" << alg.radius;
   metadata << "neighbors" << alg.neighbors;
   metadata << "grid_x" << alg.grid_x;
   metadata << "grid_y" << alg.grid_y;

   metadata << "numlabels" << (int)labels.size();
   metadata << "label_info" << "{";
   metadata << "labels" << labels;
   metadata << "numhists" << numhists;
   metadata << "}";
   metadata.release();

   return true;
}

bool ModelStorage::writeMetadata(AlgSettings alg, std::map<int, int> &labelinfo) const {
   
   // label_info
   std::vector<int> labels;
   std::vector<int> numhists;

   for(std::map<int,int>::const_iterator it = labelinfo.begin(); it != labelinfo.end(); it++) {
      labels.push_back(it->first);
      numhists.push_back(it->second);
   }

   return writeMetadata(alg, labels, numhists);
} 

//------------------------------------------------------------------------------
// Model Reading 
//------------------------------------------------------------------------------

/*
AlgSettings ModelStorage::loadAlgSettings() {
    
   FileStorage metadata(getMetadataFile(), FileStorage::READ);
   if(!metadata.isOpened()) {
      CV_Error(Error::StsError, "File '"+getMetadataFile()+"' can't be opened for reading!");
   }

   AlgSettings alg;

   metadata["radius"] >> alg.radius;
   metadata["neighbors"] >> alg.neighbors;
   metadata["grid_x"] >> alg.grid_x;
   metadata["grid_y"] >> alg.grid_y;

   setAlgSettings(alg);

   metadata.release();
   
   return alg;
} 
*/

/*
bool ModelStorage::loadLabelInfo(std::map<int,int> &labelinfo) const {
    
   FileStorage metadata(getMetadataFile(), FileStorage::READ);
   if(!metadata.isOpened()) {
      CV_Error(Error::StsError, "File '"+getMetadataFile()+"' can't be opened for reading!");
      return false;
   }

   std::vector<int> labels;
   std::vector<int> numhists;
   FileNode label_info = metadata["label_info"];
   label_info["labels"] >> labels;
   label_info["numhists"] >> numhists;

   CV_Assert(labels.size() == numhists.size());
   for(size_t idx = 0; idx < labels.size(); idx++) {
     labelinfo[labels.at((int)idx)] = numhists.at((int)idx);
   }

   metadata.release();
   return true;
} 
*/

void ModelStorage::loadMetadata(AlgSettings &alg, std::map<int,int> &labelinfo) {
 
   if(!fileExists(getMetadataFile())) {
      CV_Error(Error::StsError, "File '"+getMetadataFile()+"' doesn't exist; malformed model!");
   } 

   FileStorage metadata(getMetadataFile(), FileStorage::READ);
   if(!metadata.isOpened()) {
      CV_Error(Error::StsError, "File '"+getMetadataFile()+"' can't be opened for reading!");
   }
   
   // load alg settings
   metadata["radius"] >> alg.radius;
   metadata["neighbors"] >> alg.neighbors;
   metadata["grid_x"] >> alg.grid_x;
   metadata["grid_y"] >> alg.grid_y;

   setAlgSettings(alg);

   // load label info
   std::vector<int> labels;
   std::vector<int> numhists;
   FileNode label_info = metadata["label_info"];
   label_info["labels"] >> labels;
   label_info["numhists"] >> numhists;

   CV_Assert(labels.size() == numhists.size());
   for(size_t idx = 0; idx < labels.size(); idx++) {
     labelinfo[labels.at((int)idx)] = numhists.at((int)idx);
   }

   metadata.release();
} 

//------------------------------------------------------------------------------
// Model Information Function
//------------------------------------------------------------------------------
bool ModelStorage::checkModel(const String &name, const String &path) const {
   printf("Checking model directory at \"%s\"...\n", path.c_str());
   std::vector<String> contents = listdir(path);
   bool check = true;
   for(String file : contents) {

      if(strstr(getFileName(file).c_str(), name.c_str()) == NULL) {
         check = false;
      }
      else if(isDirectory(file)) {
         check = checkModel(name, file);
      } 

      if(!check)
         break;
   }
   return check;
} 

// A valid model is one that exists and is structured properly 
bool ModelStorage::isValidModel() const {
   if(!exists()) 
      return false;
   
   // checks that all files in the model are prefixed with the model name
   if(!checkModel(_modelname, _modelpath)) {
      return false; 
   } 

   // we have to have a labels directory, even if it's empty
   if(!isDirectory(getLabelsDir())) {
      return false; 
   } 

   // we have to have a metadata file, without it model is useless
   if(!isRegularFile(getMetadataFile())) {
      return false; 
   } 

   return true; 
} 

// Returns true if the model already exists at it's _modelpath
bool ModelStorage::exists() const {
   return fileExists(_modelpath);
} 
    
// Returns the model's path
String ModelStorage::getPath() const {
   return _modelpath; 
} 

// Returns the model's name
String ModelStorage::getName() const {
   return _modelname; 
}

AlgSettings ModelStorage::getAlgSettings() const {
   return _alg;
}

// Returns the size of the histograms in this model
int ModelStorage::getHistogramSize() const {
   return (int)(std::pow(2.0, static_cast<double>(_alg.neighbors)) * _alg.grid_x * _alg.grid_y);
} 

//------------------------------------------------------------------------------
// Model File Getters Functions - NEW
//------------------------------------------------------------------------------
String ModelStorage::intToString(int num) const {
    char numstr[16];
    sprintf(numstr, "%d", num);
    return numstr;
} 

String ModelStorage::getLabelFilePrefix(int label) const {
   return getName() + "-label-" + intToString(label);
} 

String ModelStorage::getMetadataFile() const {
   return getPath() + "/" + getName() + ".yml";
} 

String ModelStorage::getLabelsDir() const {
   return getPath() + "/" + getName() + "-labels";
}

String ModelStorage::getLabelAveragesFile() const {
   return getLabelsDir() + "/" + getName() + "-label-averages.bin";
} 

String ModelStorage::getLabelDir(int label) const {
   return getLabelsDir() + "/" + getLabelFilePrefix(label);
} 

String ModelStorage::getLabelHistogramsFile(int label) const {
   return getLabelDir(label) + "/" + getLabelFilePrefix(label) + "-histograms.bin";
} 

String ModelStorage::getLabelClusterAveragesFile(int label) const {
   return getLabelDir(label) + "/" + getLabelFilePrefix(label) + "-cluster-averages.bin";
} 

String ModelStorage::getLabelClustersFile(int label) const {
   return getLabelDir(label) + "/" + getLabelFilePrefix(label) + "-clusters.yml";
} 


//------------------------------------------------------------------------------
// Histogram Read/Write 
//------------------------------------------------------------------------------

// Wrapper functions for load/save/updating histograms for specific labels
bool ModelStorage::loadLabelHistograms(int label, std::vector<Mat> &histograms) const {
    return readHistograms(getLabelHistogramsFile(label), histograms);
}

bool ModelStorage::saveLabelHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getLabelHistogramsFile(label), histograms, false);
}

bool ModelStorage::updateLabelHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getLabelHistogramsFile(label), histograms, true);
}


// Main read/write functions for histograms
bool ModelStorage::readHistograms(const String &filename, std::vector<Mat> &histograms) const {
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


bool ModelStorage::writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const {
    FILE *fp = fopen(filename.c_str(), (appendhists == true ? "a" : "w"));
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }

    float* buffer = new float[getHistogramSize() * (int)histograms.size()];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        float* writeptr = buffer + ((int)sampleIdx * getHistogramSize());
        memcpy(writeptr, histograms.at((int)sampleIdx).ptr<float>(), getHistogramSize() * sizeof(float));
    }
    fwrite(buffer, sizeof(float), getHistogramSize() * (int)histograms.size(), fp);
    delete buffer;

    fclose(fp);
    return true;
}

//------------------------------------------------------------------------------
// Memory Mapping Histograms 
//------------------------------------------------------------------------------

void ModelStorage::mmapLabelHistograms(const std::map<int,int> &labelinfo, std::map<int, std::vector<Mat>> &histograms) const {
    
    std::cout << "loading histograms...\n";
    histograms.clear();
    for(std::map<int, int>::const_iterator it = labelinfo.begin(); it != labelinfo.end(); ++it) {
        // map histogram
        String filename = getLabelHistogramFile(it->first);
        int fd = open(filename.c_str(), O_RDONLY);
        if(fd < 0)
            CV_Error(Error::StsError, "Cannot open histogram file '"+filename+"'");

        unsigned char* mapPtr = (unsigned char*)mmap(NULL, getHistogramSize() * it->second * SIZEOF_CV_32FC1, PROT_READ, MAP_PRIVATE, fd, 0);
        if(mapPtr == MAP_FAILED)
            CV_Error(Error::StsError, "Cannot mem map file '"+filename+"'");

        // make matricies
        for(int i = 0; i < it->second; i++) {
            Mat mat(1, getHistogramSize(), CV_32FC1, mapPtr + (getHistogramSize() * SIZEOF_CV_32FC1 * i));
            histograms[it->first].push_back(mat);
        }
    }

} 

void ModelStorage::mmapLabelAverages(const std::map<int,int> &labelinfo, std::map<int, Mat> &histavgs) const {
    
    std::cout << "loading histogram averages...\n";
    histavgs.clear();
    String filename = getLabelAveragesFile();
    int fd = open(filename.c_str(), O_RDONLY);
    if(fd < 0)
        CV_Error(Error::StsError, "Cannot open histogram file '"+filename+"'");

    unsigned char* mapPtr = (unsigned char*)mmap(NULL, getHistogramSize() * (int)labelinfo.size() * SIZEOF_CV_32FC1, PROT_READ, MAP_PRIVATE, fd, 0);
    if(mapPtr == MAP_FAILED)
        CV_Error(Error::StsError, "Cannot mem map file '"+filename+"'");
    
    int idx = 0;
    for(std::map<int, int>::const_iterator it = labelinfo.begin(); it != labelinfo.end(); ++it) {
        Mat mat(1, getHistogramSize(), CV_32FC1, mapPtr + (getHistogramSize() * SIZEOF_CV_32FC1 * idx));
        histavgs[it->first] = mat;
        idx++;
    }
} 


//------------------------------------------------------------------------------
// ModelStorage Test Function
//------------------------------------------------------------------------------


void ModelStorage::test() const {
   printf("== Testing Model Storage Utility Functions ==\n");
   
   // test objects
   std::vector<String> contents;
   String testdir1 = "/dd-data/dataset/dd-dataset/dd-dataset-2"; // without trailing '/', ok
   String testdir2 = "/dd-data/dataset/dd-dataset/dd-dataset-2"; // with trailing '/', ok
   String testfile1 = "/dd-data/dataset/dd-dataset/dd-dataset-2/dd-dataset-2.tsv"; // without trailing '/', ok
   String testfile2 = "/dd-data/dataset/dd-dataset/dd-dataset-2/dd-dataset-2.tsv/"; // with trailing '/', bad
   String testsimple = "mytestfile.txt"; // not a full path, just a file name (doesn't exists though)
   String testatroot = "/mytestfile.txt"; // file directoy under root
   String testbad = "/dd-data/dataset/dd-dataset/dd-dataset-bad"; // file doesn't exist
   String testempty = ""; // empty string
   String testmkdir = "/dd-data/mkdir-parent/mkdir-child";
   String testrmrdir = "/dd-data/rmr-parent/rmr-child";

   printf(" - isDirectory\n");
   printf("For \"%s\" Expects true : %s\n", testdir1.c_str(), (isDirectory(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testdir2.c_str(), (isDirectory(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testfile1.c_str(), (isDirectory(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testfile2.c_str(), (isDirectory(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (isDirectory(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testatroot.c_str(), (isDirectory(testatroot)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (isDirectory(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (isDirectory(testempty)) ? "true" : "false");
   printf("\n");
   
   printf(" - isRegularFile\n");
   printf("For \"%s\" Expects false : %s\n", testdir1.c_str(), (isRegularFile(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testdir2.c_str(), (isRegularFile(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testfile1.c_str(), (isRegularFile(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testfile2.c_str(), (isRegularFile(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (isRegularFile(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testatroot.c_str(), (isRegularFile(testatroot)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (isRegularFile(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (isRegularFile(testempty)) ? "true" : "false");
   printf("\n");

   printf(" - fileExists\n");
   printf("For \"%s\" Expects true : %s\n", testdir1.c_str(), (fileExists(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testdir2.c_str(), (fileExists(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testfile1.c_str(), (fileExists(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false (true might be ok) : %s\n", testfile2.c_str(), (fileExists(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (fileExists(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testatroot.c_str(), (fileExists(testatroot)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (fileExists(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (fileExists(testempty)) ? "true" : "false");
   printf("\n");

   printf(" - getFileName\n");
   printf("For \"%s\" Expects \"dd-dataset-2\" : %s\n", testdir1.c_str(), getFileName(testdir1).c_str());
   printf("For \"%s\" Expects \"dd-dataset-2\" : %s\n", testdir2.c_str(), getFileName(testdir2).c_str());
   printf("For \"%s\" Expects \"dd-dataset-2.tsv\" : %s\n", testfile1.c_str(), getFileName(testfile1).c_str());
   printf("For \"%s\" Expects \"dd-dataset-2.tsv\" : %s\n", testfile2.c_str(), getFileName(testfile2).c_str());
   printf("For \"%s\" Expects \"mytestfile.txt\" : %s\n", testsimple.c_str(), getFileName(testsimple).c_str());
   printf("For \"%s\" Expects \"mytestfile.txt\" : %s\n", testatroot.c_str(), getFileName(testatroot).c_str());
   printf("For \"%s\" Expects \"dd-dataset-bad\" : %s\n", testbad.c_str(), getFileName(testbad).c_str());
   printf("For \"%s\" Expects \"\" : %s\n", testempty.c_str(), getFileName(testempty).c_str());
   printf("\n");

   printf(" - getFileParent\n");
   printf("For \"%s\" Expects \"/dd-data/dataset/dd-dataset\" : %s\n", testdir1.c_str(), getFileParent(testdir1).c_str());
   printf("For \"%s\" Expects \"/dd-data/dataset/dd-dataset\" : %s\n", testdir2.c_str(), getFileParent(testdir2).c_str());
   printf("For \"%s\" Expects \"/dd-data/dataset/dd-dataset/dd-dataset-2\" : %s\n", testfile1.c_str(), getFileParent(testfile1).c_str());
   printf("For \"%s\" Expects \"/dd-data/dataset/dd-dataset/dd-dataset-2\" : %s\n", testfile2.c_str(), getFileParent(testfile2).c_str());
   printf("For \"%s\" Expects \"\" : %s\n", testsimple.c_str(), getFileParent(testsimple).c_str());
   printf("For \"%s\" Expects \"/\" : %s\n", testatroot.c_str(), getFileParent(testatroot).c_str());
   printf("For \"%s\" Expects \"/dd-data/dataset/dd-dataset/dd-dataset-bad\" : %s\n", testbad.c_str(), getFileParent(testbad).c_str());
   printf("For \"%s\" Expects \"\" : %s\n", testempty.c_str(), getFileParent(testempty).c_str());
   printf("\n");

   printf(" - mkdirs\n");
   printf("For \"%s\" Expects true : %s\n", testmkdir.c_str(), (mkdirs(testmkdir)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testrmrdir.c_str(), (mkdirs(testrmrdir)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (mkdirs(testempty)) ? "true" : "false");
   printf("\n");

   printf(" - rmr\n");
   printf("For \"%s\" Expects true : %s\n", testrmrdir.c_str(), (rmr(testrmrdir)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (rmr(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (rmr(testempty)) ? "true" : "false");
   printf("\n");

   system(("rm -r " + testmkdir).c_str());
   system(("rm -r " + testrmrdir).c_str());

   printf(" - listdir\n");
   contents = listdir(testdir1);
   printf("Contents of \"%s\" (should be 7 items):\n", testdir1.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testdir2);
   printf("Contents of \"%s\" (should be 7 items):\n", testdir2.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testfile1);
   printf("Contents of \"%s\" (should be empty):\n", testfile1.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testfile2);
   printf("Contents of \"%s\" (should be empty):\n", testfile2.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testsimple);
   printf("Contents of \"%s\" (should be empty):\n", testsimple.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testbad);
   printf("Contents of \"%s\" (should be empty):\n", testbad.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testempty);
   printf("Contents of \"%s\" (should be empty):\n", testempty.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   
   printf(" - getAlgSettings\n");
   AlgSettings alg = getAlgSettings();
   printf("radius = %d\n", alg.radius);
   printf("neighbors = %d\n", alg.neighbors);
   printf("grid_x = %d\n", alg.grid_x);
   printf("grid_y = %d\n", alg.grid_y);
   printf("\n");

   printf(" - getHistogramSize\n");
   printf("histsize = %d\n", getHistogramSize());

   printf("\n");
   printf(" !! End of Utility Functions Tests !!\n");
   printf("\n");
} 


}}
