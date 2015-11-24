set(the_description "Digital Democracy Facial Recognition")
ocv_define_module(ddface opencv_face opencv_core opencv_imgproc opencv_objdetect WRAP python)
SET(CMAKE_CXX_FLAGS "-std=c++0x -pthread -Wl,-rpath,./libs/tbb_release -L./libs/tbb_release")
# NOTE: objdetect module is needed for one of the samples
