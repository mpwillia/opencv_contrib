set(the_description "Digital Democracy Facial Recognition")
ocv_define_module(ddface opencv_face opencv_core opencv_imgproc opencv_objdetect WRAP python)
SET(CMAKE_CXX_FLAGS "-std=c++0x -pthread -I~/Development/dd-FacialRecognition/tbb/include -Wl,-rpath,~/Development/dd-FacialRecognition/tbb/release -L~/Development/dd-FacialRecognition/tbb/release -ltbb")
# NOTE: objdetect module is needed for one of the samples
