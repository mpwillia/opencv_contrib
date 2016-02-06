
#include "modelstorage.hpp"

#include <cstring>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>

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
   if(!fileExists(parent)) {
      // our parent doesn't exist, so lets try to make it
      if(!mkdirs(parent)) {
         // if we failed to make our parent, return false
         return false;
      } 
   } 

   if(isDirectory(parent)) {
      // good, our parent is a directory, so lets try to make ourselves
      // return true if we get no errors making it; false otherwise
      return (mkdir(dirpath.c_str(), DEFFILEMODE) == 0);
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
         if(!rmr(file)) 
            return false;
      } 
   } 
   
   return (remove(filepath.c_str()) == 0);
} 

//------------------------------------------------------------------------------
// ModelStorage Functions 
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
   printf("For \"%s\" Expects true : \n", testmkdir.c_str(), (mkdirs(testmkdir)) ? "true" : "false");
   printf("For \"%s\" Expects true : \n", testrmrdir.c_str(), (mkdirs(testrmrdir)) ? "true" : "false");
   printf("For \"%s\" Expects true : \n", testempty.c_str(), (mkdirs(testempty)) ? "true" : "false");

   printf(" - rmr\n");
   printf("For \"%s\" Expects true : \n", testrmrdir.c_str(), (rmr(testrmrdir)) ? "true" : "false");
   printf("For \"%s\" Expects false : \n", testbad.c_str(), (rmr(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : \n", testempty.c_str(), (rmr(testempty)) ? "true" : "false");

   printf(" - listdir\n");
   contents = listdir(testdir1);
   printf("Contents of \"%s\" (should be 7 items):", testdir1.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testdir2);
   printf("Contents of \"%s\" (should be 7 items):", testdir2.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testfile1);
   printf("Contents of \"%s\" (should be empty):", testfile1.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testfile2);
   printf("Contents of \"%s\" (should be empty):", testfile2.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testsimple);
   printf("Contents of \"%s\" (should be empty):", testsimple.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testbad);
   printf("Contents of \"%s\" (should be empty):", testbad.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   contents = listdir(testempty);
   printf("Contents of \"%s\" (should be empty):", testempty.c_str());
   for(String s : contents)
      printf("\t\"%s\"\n", s.c_str());
   printf("\n");

   printf("\n");
   printf(" !! End of Utility Functions Tests !!\n");
   printf("\n");
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
      }

   }

   return true;
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
   
   return checkModel(_modelname, _modelpath);
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


//------------------------------------------------------------------------------
// Model File Getters Functions 
//------------------------------------------------------------------------------
String ModelStorage::getInfoFile() const {
   return getPath() + "/" + getName() + ".yml";
} 

String ModelStorage::getHistogramsDir() const {
   return getPath() + "/" + getName() + "-histograms";
} 

String ModelStorage::getHistogramFile(int label) const {
    char labelstr[16];
    sprintf(labelstr, "%d", label);
    return getHistogramsDir() + "/" + getName() + "-" + labelstr + ".bin";
} 

String ModelStorage::getHistogramAveragesFile() const {
    return getHistogramsDir() + "/" + getName() + "-averages.bin";
} 


//------------------------------------------------------------------------------
// Histogram Read/Write 
//------------------------------------------------------------------------------

// Wrapper functions for load/save/updating histograms for specific labels
bool ModelStorage::loadHistograms(int label, std::vector<Mat> &histograms, int histSize) const {
    return readHistograms(getHistogramFile(label), histograms, histSize);
}

bool ModelStorage::saveHistograms(int label, const std::vector<Mat> &histograms, int histSize) const {
    return writeHistograms(getHistogramFile(label), histograms, false, histSize);
}

bool ModelStorage::updateHistograms(int label, const std::vector<Mat> &histograms, int histSize) const {
    return writeHistograms(getHistogramFile(label), histograms, true, histSize);
}


// Main read/write functions for histograms
bool ModelStorage::readHistograms(const String &filename, std::vector<Mat> &histograms, int histSize) const {
    FILE *fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    
    float buffer[histSize];
    while(fread(buffer, sizeof(float), histSize, fp) > 0) {
        Mat hist = Mat::zeros(1, histSize, CV_32FC1);
        memcpy(hist.ptr<float>(), buffer, histSize * sizeof(float));
        histograms.push_back(hist);
    }
    fclose(fp);
    return true;
}


bool ModelStorage::writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists, int histSize) const {
    FILE *fp = fopen(filename.c_str(), (appendhists == true ? "a" : "w"));
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }

    float* buffer = new float[histSize * (int)histograms.size()];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        float* writeptr = buffer + ((int)sampleIdx * histSize);
        memcpy(writeptr, histograms.at((int)sampleIdx).ptr<float>(), histSize * sizeof(float));
    }
    fwrite(buffer, sizeof(float), histSize * (int)histograms.size(), fp);
    delete buffer;

    fclose(fp);
    return true;
}


}}
