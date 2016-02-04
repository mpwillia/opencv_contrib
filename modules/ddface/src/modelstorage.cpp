
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
bool isDirectory(const String &filepath) {
   struct stat buffer;
   if(stat(filepath.c_str(), &buffer)==0)
      return S_ISDIR(buffer.st_mode);
   else
      return false;
} 

// Returns true if the file at the given path is a regular file; false otherwise
bool isRegularFile(const String &filepath) {
   struct stat buffer;
   if(stat(filepath.c_str(), &buffer)==0)
      return S_ISREG(buffer.st_mode);
   else
      return false;
} 

// Returns true if the file at the given path exists; false otherwise
bool exists(const String &filepath) {
   struct stat buffer;     
   return (stat(filepath.c_str(), &buffer) == 0);
} 

// Gets the name of the file from the given filepath
String getFileName(const String &filepath) {
   size_t idx = filepath.find_last_of('/');

   if((int)idx < 0) {
      // if we don't find a '/' at all then just return the path
      return filepath;
   }
   else if((int)idx >= (int)filepath.length()-1) {
      // if we have a trailing '/' then remove it and try again
      return getFileName(filepath.substr(0, path.length()-1));
   }
   return filepath.substr(idx + 1);
} 

// Returns filepaths to all of the files within the given directory
// Assumes the given path is a valid existing directory with permissions
// If the given path isn't a directory or it doesn't exist a empty vectory will be returned
std::vector<String> listdir(const String &dirpath) {
   std::vector<String> contents;
   if(!isDirectory(dirpath))
      return contents;

   struct dirent **namelist;
   int n;

   bool check = true;
   n = scandir(dirpath, &namelist, NULL, alphasort);
   if (n < 0)
      CV_Error(Error::StsError, "Error reading directory at '"+dirpath+"'"); 
   else {
      while (n--) {
         contents.push_back(String(namelist[n]->d_name));
         free(namelist[n]);
      }
      free(namelist);
   }
   return contents;
} 

//------------------------------------------------------------------------------
// ModelStorage Functions 
//------------------------------------------------------------------------------


void ModelStorage::test() const {
   printf("== Testing Model Storage Utility Functions ==\n");
   
   // test objects
   std::vector<String> contents;
   String testdir1 = "/dd-data/dataset/dd-dataset"; // without trailing '/', ok
   String testdir2 = "/dd-data/dataset/dd-dataset/"; // with trailing '/', ok
   String testfile1 = "/dd-data/dataset/dd-dataset/dd-dataset-2/dd-dataset-2.tsv"; // without trailing '/', ok
   String testfile2 = "/dd-data/dataset/dd-dataset/dd-dataset-2/dd-dataset-2.tsv/"; // with trailing '/', bad
   String testsimple = "mytestfile.txt"; // not a full path, just a file name (doesn't exists though)
   String testbad = "/dd-data/dataset/dd-dataset/dd-dataset-bad"; // file doesn't exist
   String testempty = ""; // empty string

   printf(" - isDirectory\n");
   printf("For \"%s\" Expects true : %s\n", testdir1.c_str(), (isDirectory(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testdir2.c_str(), (isDirectory(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testfile1.c_str(), (isDirectory(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testfile2.c_str(), (isDirectory(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (isDirectory(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (isDirectory(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (isDirectory(testempty)) ? "true" : "false");
   print("\n");

   printf(" - isRegularFile\n");
   printf("For \"%s\" Expects false : %s\n", testdir1.c_str(), (isRegularFile(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testdir2.c_str(), (isRegularFile(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testfile1.c_str(), (isRegularFile(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false (true might be ok) : %s\n", testfile2.c_str(), (isRegularFile(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (isRegularFile(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (isRegularFile(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (isRegularFile(testempty)) ? "true" : "false");
   print("\n");

   printf(" - exists\n");
   printf("For \"%s\" Expects true : %s\n", testdir1.c_str(), (exists(testdir1)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testdir2.c_str(), (exists(testdir2)) ? "true" : "false");
   printf("For \"%s\" Expects true : %s\n", testfile1.c_str(), (exists(testfile1)) ? "true" : "false");
   printf("For \"%s\" Expects false (true might be ok) : %s\n", testfile2.c_str(), (exists(testfile2)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testsimple.c_str(), (exists(testsimple)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testbad.c_str(), (exists(testbad)) ? "true" : "false");
   printf("For \"%s\" Expects false : %s\n", testempty.c_str(), (exists(testempty)) ? "true" : "false");
   print("\n");

   printf(" - getFileName\n");
   printf("For \"%s\" Expects \"dd-dataset\" : %s\n", testdir1.c_str(), getFileName(testdir1));
   printf("For \"%s\" Expects \"dd-dataset\" : %s\n", testdir2.c_str(), getFileName(testdir2));
   printf("For \"%s\" Expects \"dd-dataset-2.tsv\" : %s\n", testfile1.c_str(), getFileName(testfile1));
   printf("For \"%s\" Expects \"dd-dataset-2.tsv\" : %s\n", testfile2.c_str(), getFileName(testfile2));
   printf("For \"%s\" Expects \"mytestfile.txt\" : %s\n", testsimple.c_str(), getFileName(testsimple));
   printf("For \"%s\" Expects \"dd-dataset-bad\" : %s\n", testbad.c_str(), getFileName(testbad));
   printf("For \"%s\" Expects \"\" : %s\n", testempty.c_str(), getFileName(testempty));
   print("\n");

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

// A valid model is one that exists and is structured properly 
bool ModelStrong::isValidModel() const {
   if(!modelExists()) 
      return false;

   // check model structure
   struct dirent **namelist;
   int n;

   bool check = true;
   n = scandir(_modelpath, &namelist, NULL, alphasort);
   if (n < 0)
      CV_Error(Error::StsError, "Error reading directory at '"+_modelpath+"'"); 
   else {
      while (n--) {
         printf("%s\n", namelist[n]->d_name);

         if(strstr(namelist[n]->d_name, _modelname) == NULL)
            check = false;

         free(namelist[n]);

         if(!check)
            break;
      }
      free(namelist);
   }

   return check;
} 

// Returns true if the model already exists at it's _modelpath
bool ModelStorage::modelExists() const {
   return exists(_modelpath);
} 

// Returns the model's path
String ModelStorage::getModelPath() const {
   return _modelpath; 
} 

// Returns the model's name
String ModelStorage::getModelName() const {
   return _modelname; 
}

}}
