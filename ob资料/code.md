- 登录系统租户
obclient -h 127.0.0.1 -P2881 -uroot

- 测试租户
obclient -h 127.0.0.1 -P2881 -uroot@perf

- trace 
set ob_enable_show_trace=1;
show trace;

- 日志
show variables like "%ob_log_level%";
set ob_log_level=debug;
set ob_log_level=info;
set ob_log_level=disabled;

alter system set syslog_io_bandwidth_limit='2G';
alter system set enable_async_syslog='False';

- perf
scp 192.168.20.103:/home/jokerjay/source/FlameGraph/perf.folded .
scp -J player_6@47.242.130.84 root@192.168.0.242:/root/source/FlameGraph/perf.folded .

- 向量辅助表
set ob_enable_index_direct_select = 1;
select tenant_id, table_id, table_name, tablet_id, data_table_id from oceanbase.__all_virtual_table where table_name like '%5000%';

- 栈
addr2line -pCfe /data/obcluster/bin/observer 0x1ba1edc0 0xf2250a0 0xf00d4f2 0xdc701d4 0xdd6abb5 0xdb77f85 0x1112671b 0x11123647 0xdc4c997 0xdc4d966 0xdc4d45a 0xb9cfa0b 0xba1cf5b 0xba184c5 0x1cba626a 0xb7b214d 0xb7b1936 0xb7b0380 0xb7af791 0xb7b0a47 0x1b461663 0x7fdc79a081ca 0x7fdc796398d3

addr2line -pCfe /home/jokerjay/data/obcluster/bin/observer 0x1bddfac8 0x1b568a1c 0x7eff97412d0f 0x13440e71 0x13434c7d 0x1345a9e1 0x1342052f 0x133b2bbe 0x10bc29f3 0x109d9c18 0x109dc4ad 0x109db724 0xecbaccc 0x109dae59 0xecb753e 0xec4abd4 0xeca9360 0x109d6c6d 0x105a0b49 0x1059ff65 0x1059e038 0x1059eb3f 0x105e2d02 0xb770729 0xb770df0 0xb7709fa 0x1b46c872 0x7eff974081c9 0x7eff970398d2
- 计时
std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
auto finish = std::chrono::steady_clock::now();
std::chrono::duration<double, std::milli> duration = finish - start;
LOG_INFO("ChenNingjie: open result",K(duration.count()));



use test;
set ob_log_level=debug;
CREATE TABLE t2 (id int,c1 int, embedding vector(3), primary key(id), key(c1));
insert into t2 values(2,2,[2,1,1]);

create vector index idx1 on t2(embedding) with (distance=l2, type=hnsw, lib=vsag, m=16, ef_construction=200);

drop index idx1 on t2;


1. 调整了M、ef_construction、ef_search参数
2. vid由id左移32位+c1构成，通过该关系实现了越过AUX表、in-filter
3. 对于没有过滤条件的情况，直接返回id，越过了主表
4. 取消了“增量”部分的代码、包括标志删除的bitmap等
5. 采用单例工作线程解析向量的json。实现了并行化
6. 减少了searchbaselayer的分支判断(与增量部分相关的)。增加了超参(循环轮数)提前剪枝。