#!/bin/bash

# Force reinitialize oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

# Set Level Zero environment variables
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=0

# Clear any existing device filters
unset SYCL_DEVICE_FILTER
unset SYCL_PI_TRACE
unset SYCL_UR_TRACE
