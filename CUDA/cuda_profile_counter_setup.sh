#! /bin/bash 

# (1) CUDA_PROFILE: Set to 1 or 0 to enable/disable the profiler
export CUDA_PROFILE="0";
echo "CUDA_PROFILE:" $CUDA_PROFILE;

# (2) CUDA_PROFILE_LOG: Set to the name of the log file (The default is ./cuda_profile.log)
export CUDA_PROFILE_LOG=${PWD}/cuda_profile.log
echo "CUDA_PROFILE_LOG:" $CUDA_PROFILE_LOG;

# (3) CUDA_PROFILE_CVS: Set to 1 or 0 to enable or disable a comma separated version of the log

export CUDA_PROFILE_CVS=1
# (4) Specify a configuration file with up to four signals
# here . belongs to the name ".cuda_profile_config"
export CUDA_PROFILE_CONFIG="${PWD}/.cuda_profile_config"
echo "CUDA_PROFILE_CONFIG:"  $CUDA_PROFILE_CONFIG

