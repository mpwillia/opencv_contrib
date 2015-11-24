#!/bin/csh
setenv TBBROOT "/home/mpwillia/Development/dd-FacialRecognition/tbb/tbb44_20151010oss" #
setenv tbb_bin "/home/mpwillia/Development/dd-FacialRecognition/tbb/tbb44_20151010oss/build/linux_intel64_gcc_cc4.8.2_libc2.17_kernel3.14.44_release" #
if (! $?CPATH) then #
    setenv CPATH "${TBBROOT}/include" #
else #
    setenv CPATH "${TBBROOT}/include:$CPATH" #
endif #
if (! $?LIBRARY_PATH) then #
    setenv LIBRARY_PATH "${tbb_bin}" #
else #
    setenv LIBRARY_PATH "${tbb_bin}:$LIBRARY_PATH" #
endif #
if (! $?LD_LIBRARY_PATH) then #
    setenv LD_LIBRARY_PATH "${tbb_bin}" #
else #
    setenv LD_LIBRARY_PATH "${tbb_bin}:$LD_LIBRARY_PATH" #
endif #
 #
