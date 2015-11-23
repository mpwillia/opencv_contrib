set(the_description "Digital Democracy Face Recognition")
ocv_define_module(ddface face opencv_core opencv_imgproc opencv_objdetect WRAP python)
set(CMAKE_CXX_FLAGS "-std=c++0x -pthread")
# NOTE: objdetect module is needed for one of the samples
