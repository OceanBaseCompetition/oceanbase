#!/bin/bash
/root/source/oceanbase/tools/deploy/obd.sh stop -n obcluster

bash build.sh release --init --make -j3
rm -f  /root/data/obcluster/bin/observer
# cp ./build_debug/src/observer/observer /data/obcluster/bin/
cp ./build_release/src/observer/observer /root/data/obcluster/bin/

/root/source/oceanbase/tools/deploy/obd.sh start -n obcluster


sleep 13
obclient -h127.0.0.1 -P2881 -uroot@perf