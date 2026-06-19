#!/bin/bash
export LD_LIBRARY_PATH="/usr/local/gcc-14.1.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/../build/src/cpp/taosim" "$@"
