#!/bin/bash
/root/source/.oceanbase/oceanbase/tools/deploy/obd.sh stop -n obcluster
cd /root/source/.oceanbase/oceanbase/build_release && make -j50 --silent
# bash build.sh release --init --make -j3
# bash build.sh release --init --make -j3
rm -f  /data/obcluster/bin/observer
rm -f /data/obcluster/log/observer.log
cp /root/source/.oceanbase/oceanbase/build_release/src/observer/observer /data/obcluster/bin/

/root/source/.oceanbase/oceanbase/tools/deploy/obd.sh start -n obcluster


# ./deps/3rd/u01/obclient/bin/obclient -h127.0.0.1 -P2881 -uroot@perf;