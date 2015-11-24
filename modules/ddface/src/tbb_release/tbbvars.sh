#!/bin/bash
export TBBROOT="/home/mpwillia/Development/dd-FacialRecognition/tbb/tbb44_20151010oss" #
tbb_bin="/home/mpwillia/Development/dd-FacialRecognition/tbb/tbb44_20151010oss/build/linux_intel64_gcc_cc4.8.2_libc2.17_kernel3.14.44_release" #
if [ -z "$CPATH" ]; then #
    export CPATH="${TBBROOT}/include" #
else #
    export CPATH="${TBBROOT}/include:$CPATH" #
fi #
if [ -z "$LIBRARY_PATH" ]; then #
    export LIBRARY_PATH="${tbb_bin}" #
else #
    export LIBRARY_PATH="${tbb_bin}:$LIBRARY_PATH" #
fi #
if [ -z "$LD_LIBRARY_PATH" ]; then #
    export LD_LIBRARY_PATH="${tbb_bin}" #
else #
    export LD_LIBRARY_PATH="${tbb_bin}:$LD_LIBRARY_PATH" #
fi #
 #
