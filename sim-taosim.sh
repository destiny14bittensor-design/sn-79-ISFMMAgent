#!/usr/bin/env bash
# Wait for proxy to be ready before starting taosim
sleep 10
export LD_LIBRARY_PATH=/usr/local/gcc-14.1.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /home/administrator/workspace/sn-79/simulate/trading/run/../build/src/cpp/taosim \
    -f /home/administrator/workspace/sn-79/simulate/trading/run/config/simulation_0.xml
