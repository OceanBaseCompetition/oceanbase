bash build.sh debug --init --make
./tools/deploy/obd.sh stop -n single
cp build_debug/src/observer/observer /tmp/obtest/observer1/bin/observer
./tools/deploy/obd.sh start -n single