# author: Philipp Arndt, UC San Diego / Scripps Institution of Oceanography
#
# 本脚本用途：对单条 ICESat-2 ATL03 轨（granule）下载数据 → 检测融水湖 → SuRRF 反演水深 → 输出 h5/jpg/csv
# 在 OSG 集群上由 run_py.sh 调用；本地学习时可单独运行（见下方示例命令）
#
# 本地运行示例：
#   conda activate eeicelakes-env
#   python detect_lakes.py --granule ATL03_20220603025742_11001503_007_01.h5 --polygon geojsons/simplified_GRE_2000_CW.geojson

import argparse      # 解析命令行参数（--granule、--polygon 等）
import os            # 文件/目录操作（创建文件夹、删临时文件等）
import gc            # 垃圾回收，释放大数组占用的内存
import sys           # 系统退出（sys.exit）等
import time          # 下载失败时 sleep 等待后重试
import pickle        # 序列化（本脚本未直接使用，保留供集群环境兼容）
import subprocess    # 子进程（本脚本未直接使用，保留供集群环境兼容）
import traceback     # 打印完整错误堆栈，便于调试
import requests      # 识别 HTTP 401/403，避免无意义重试
import numpy as np   # 数值计算（文件名中的质量分数编码等）
import icelakes      # 项目包，确保 icelakes 模块可被导入
from icelakes.utilities import get_size          # 格式化文件大小（如 "12.3 MB"）
from icelakes.nsidc import download_granule, verify_local_granule  # 从 NSIDC/Earthdata 下载并空间子集 ATL03
from icelakes.detection import read_atl03, detect_lakes, melt_lake  # 读 h5、检测湖、melt_lake 类

# ---------- 命令行参数定义 ----------
parser = argparse.ArgumentParser(description='Test script to print some stats for a given ICESat-2 ATL03 granule.')
parser.add_argument('--granule', type=str, default='ATL03_20220714010847_03381603_006_02.h5',
                    help='The producer_id of the input ATL03 granule')  # 要处理的 ATL03 文件名（来自 granule 清单第 1 列）
parser.add_argument('--polygon', type=str, default='geojsons/simplified_GRE_2000_CW.geojson',
                    help='The file path of a geojson file for spatial subsetting')  # 研究区 simplified geojson（清单第 2 列）
parser.add_argument('--is2_data_dir', type=str, default='IS2data',
                    help='The directory into which to download ICESat-2 granules')  # 下载的 ATL03 临时存放目录
parser.add_argument('--download_gtxs', type=str, default='all',
                    help='String value or list of gtx names to download, also accepts "all"')  # 下载哪些激光束：all 或 gt1l 等
parser.add_argument('--out_data_dir', type=str, default='detection_out_data',
                    help='The directory to which to write the output data')  # 每个湖的 .h5 数据输出目录
parser.add_argument('--out_plot_dir', type=str, default='detection_out_plot',
                    help='The directory to which to write the output plots')  # 每个湖的剖面图 .jpg 输出目录
parser.add_argument('--out_stat_dir', type=str, default='detection_out_stat',
                    help='The directory to which to write the granule stats')  # 本 granule 汇总统计 .csv 输出目录
parser.add_argument('--skip-download', action='store_true',
                    help='Skip NSIDC download; use existing file in --is2_data_dir (manual download)')
parser.add_argument('--keep-granule', action='store_true',
                    help='Keep downloaded ATL03 in IS2data/ after processing (for re-runs)')
args = parser.parse_args()  # 解析命令行，得到 args.granule、args.polygon 等

print('\npython args:', args, '\n')  # 打印实际使用的参数，便于核对
print('Process PID: %i  (若 Ctrl+C 无效，请新开 PowerShell 执行: Stop-Process -Id %i -Force)\n' % (os.getpid(), os.getpid()))

if not os.path.isfile(args.polygon):
    print('Polygon file not found:', args.polygon)
    if args.polygon.endswith('python'):
        print('  (Did you paste "python" onto the end of --polygon? Use the path ending in .geojson only.)')
    sys.exit(1)

# ---------- NASA Earthdata 凭证（仅下载时需要）----------
earthdata_uid = earthdata_pwd = None
if not args.skip_download:
    try:
        from ed.edcreds import getedcreds
    except ModuleNotFoundError:
        print('Please copy ed/edcreds.example.py to ed/edcreds.py and set your NASA Earthdata credentials.')
        sys.exit(1)
    earthdata_creds = getedcreds()
    if earthdata_creds is None:
        print('Please set your NASA Earthdata username/password/email in ed/edcreds.py')
        sys.exit(1)
    earthdata_uid, earthdata_pwd, _earthdata_email = earthdata_creds

# ---------- 创建输入/输出目录（HTCondor 作业也会在节点上建这些目录）----------
print('Shuffling files around for HTCondor...')
for thispath in (args.is2_data_dir, args.out_data_dir, args.out_plot_dir, args.out_stat_dir):
    if not os.path.exists(thispath): os.makedirs(thispath)  # 目录不存在则创建

# ---------- 从 NSIDC 下载指定 granule（失败则最多重试 50 次）----------
local_h5 = os.path.join(args.is2_data_dir, args.granule)
local_ok, local_msg = verify_local_granule(local_h5, args.granule) if os.path.isfile(local_h5) else (False, '')
if args.skip_download or local_ok:
    if not os.path.isfile(local_h5):
        print('Granule not found (need manual download):', local_h5)
        print('Download in browser from https://search.earthdata.nasa.gov then place file in IS2data/')
        sys.exit(1)
    if not local_ok:
        print(local_msg)
        print('\nTo complete the download (supports resume), run:')
        print('  python download_one_granule.py')
        print('Then re-run detect_lakes.py with --skip-download\n')
        sys.exit(1)
    input_filename = local_h5
    request_status_code = 200
    print('Skipping download; using existing file: %s (%s)' % (input_filename, get_size(input_filename)))
else:
    try_nr = 1                      # 当前是第几次尝试
    request_status_code = 0         # HTTP 状态码，200 表示下载成功
    while (request_status_code != 200) & (try_nr <= 50):  # 非 200 且未超过 50 次则继续循环
        try:
            print('Downloading granule from NSIDC. (try %i)' % try_nr)
            input_filename, request_status_code = download_granule(
                args.granule,
                args.download_gtxs,
                args.polygon,
                args.is2_data_dir,
                earthdata_uid,
                earthdata_pwd,
                spatial_sub=True
            )
            if request_status_code != 200:
                if request_status_code in (401, 403):
                    print('  --> Earthdata authentication failed (%i). Fix ed/edcreds.py and NSIDC approvals; not retrying.\n' % request_status_code)
                    break
                print('  --> Request unsuccessful (%i), trying again in a minute...\n' % request_status_code)
                time.sleep(60)
                try_nr += 1

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                print('  --> Earthdata authentication failed. Not retrying.\n')
                request_status_code = e.response.status_code
                break
            print('  --> Request unsuccessful (error raised in code), trying again in a minute...\n')
            traceback.print_exc()
            time.sleep(60)
            try_nr += 1
        except KeyboardInterrupt:
            print('\nInterrupted by user.')
            sys.exit(130)
        except:
            print('  --> Request unsuccessful (error raised in code), trying again in a minute...\n')
            traceback.print_exc()
            time.sleep(60)
            try_nr += 1

# ---------- 检查下载是否成功 ----------
print('Request status code:', request_status_code, request_status_code==200)
if request_status_code == 200:
    print('NSIDC API request was successful!')
if request_status_code != 200:
    print('NSIDC API request failed. (Request status code: %i)' % request_status_code)
    sys.exit(127)                   # 退出码 127：下载失败（HTCondor 会据此标记 job 失败）
if request_status_code==200:
    with open('success.txt', 'w') as f: print('we got some sweet data', file=f)  # 集群用此文件判断“有数据”
    if input_filename == 'none':    # CMR 未找到该 granule
        print('no granule found. nothing more to do here.')
        sys.exit(69)                # 退出码 69：无数据但不算硬错误（HTCondor success_exit_code）
if os.path.exists(input_filename):
    ok, msg = verify_local_granule(input_filename, args.granule)
    if not ok:
        print(msg)
        sys.exit(127)
    if os.path.getsize(input_filename) < 1000000:  # 文件小于约 1 MB，视为空/无效
        print('granule seems to be empty. nothing more to do here.')
        sys.exit(69)

# ---------- 读取 ATL03：只取 beam 列表和辅助信息，暂不读全部光子（gtxs_to_read='none'）----------
gtx_list, ancillary = read_atl03(input_filename, gtxs_to_read='none')  # gtx_list: 如 ['gt1l','gt1r',...]

# ---------- 对每条 ground track（beam）检测融水湖 ----------
lake_list = []                  # 本 granule 检出的所有湖（melt_lake 对象列表）
granule_stats = [0,0,0,0]       # 累计统计：[轨迹总长度, 湖段长度, 光子总数, 湖内光子数]

for gtx in gtx_list:            # 遍历 6 条 beam（gt1l, gt1r, gt2l, gt2r, gt3l, gt3r）
    try:
        lakes_found, gtx_stats = detect_lakes(input_filename, gtx, args.polygon, verbose=False)
        # lakes_found: 该 beam 上检出的湖列表；gtx_stats: 该 beam 的 4 项统计
        for i in range(len(granule_stats)): granule_stats[i] += gtx_stats[i]  # 累加到 granule 级统计
        lake_list += lakes_found    # 合并到总湖列表
        del lakes_found, gtx_stats  # 释放临时变量
        gc.collect()                # 主动触发垃圾回收（大 granule 时防内存爆）
    except:
        print('Something went wrong for %s' % gtx)
        traceback.print_exc()       # 单 beam 失败不影响其它 beam

try:
    if granule_stats[0] > 0:      # 若轨迹长度 > 0，说明有可用光子数据
        with open('success.txt', 'w') as f: print('we got some useable data from NSIDC!!', file=f)
        print('Sucessfully got some useable data from NSIDC!!')
except:
    traceback.print_exc()

# 打印本 granule 汇总统计
try:
    print('\nDetected %i lakes before length filter.' % len(lake_list))
    print('GRANULE STATS (length total, length lakes, photons total, photons lakes):%.3f,%.3f,%i,%i\n' % tuple(granule_stats))
except:
    traceback.print_exc()

# ---------- 过滤异常长的“湖”（>50 km 多为海洋等误检；格陵兰融水带可较长）----------
try:
    max_lake_length = 50000       # 单位：米（原 20 km + photon_data 度量曾误删全部湖）
    n_before = len(lake_list)
    # 用实际水面长度 length_water_surfaces，不要用 photon_data 的 xatc 跨度
    # （photon_data 含整段 mframe 窗口，常 >20 km，会把所有湖误删）
    lake_list[:] = [lake for lake in lake_list if lake.length_water_surfaces <= max_lake_length]
    if n_before > 0 and len(lake_list) == 0:
        print('Note: %i lakes detected but all removed by >%i m length filter.' % (n_before, max_lake_length))
    elif n_before != len(lake_list):
        print('Lake filter: %i -> %i (removed segments > %i m)' % (n_before, len(lake_list), max_lake_length))
except:
    traceback.print_exc()

# ---------- SuRRF：对每个湖反演水深 ----------
print('---> determining depth for each lake')
for i, lake in enumerate(lake_list):
    try:
        lake.surrf()                # 稳健非参数回归拟合水面/湖床，得到 max_depth、lake_quality 等
        print('   --> %3i/%3i, %s | %8.3fN, %8.3fE: %6.2fm deep / quality: %8.2f' % (i+1, len(lake_list), lake.gtx, lake.lat,
                                                                                 lake.lon, lake.max_depth, lake.lake_quality))
    except:
        print('Error for lake %i (detection quality = %.5f) ... skipping:' % (i+1, lake.detection_quality))
        traceback.print_exc()
        lake.lake_quality = 0.0     # 水深反演失败则质量置 0

# 可选：去掉 lake_quality 为 0 的湖（当前被注释掉，失败湖仍会尝试出图）
# lake_list[:] = [lake for lake in lake_list if lake.lake_quality > 0]

# ---------- 为每个湖生成 ID、出图、写 h5 ----------
for i, lake in enumerate(lake_list):
    try:
        # 唯一湖 ID：流域名_granule前缀_beam_序号
        lake.lake_id = '%s_%s_%s_%04i' % (lake.polygon_name, lake.granule_id[:-3], lake.gtx, i)
        # 文件名嵌入质量分数，便于按质量排序（数字越小通常质量越好）
        filename_base = 'lake_%05i-%05i_%s_%s_%s' % (np.clip(1000-lake.lake_quality,0,None)*10,
                                                     np.clip(1.0-lake.detection_quality,0,1)*1e4,
                                                     lake.ice_sheet, lake.melt_season,
                                                     lake.lake_id)
        figname = os.path.join(args.out_plot_dir, '%s.jpg' % filename_base)
        h5name = os.path.join(args.out_data_dir, '%s.h5' % filename_base)

        # 绘制光子剖面 + 水面/湖床拟合线，保存 jpg
        try:
            fig = lake.plot_lake(closefig=True)
            if fig is not None: fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
        except:
            print('Could not make figure for lake <%s>' % lake.lake_id)
            traceback.print_exc()

        # 将湖对象写入 HDF5
        try:
            datafile = lake.write_to_hdf5(h5name)
            print('Wrote data file: %s, %s' % (datafile, get_size(datafile)))
        except:
            print('Could not write hdf5 file <%s>' % lake.lake_id)
            traceback.print_exc()

        # 若只成功一半：保留 h5（数据更重要）；仅在没有 h5 时删 jpg
        if os.path.isfile(figname) and (not os.path.isfile(h5name)):
            os.remove(figname)
        # 不再删除仅有 h5 无 jpg 的文件

    except:
        traceback.print_exc()

# ---------- 写本 granule 的统计 csv（一行）----------
try:
    statsfname = args.out_stat_dir + '/stats_%s_%s.csv' % (args.polygon[args.polygon.rfind('/')+1:].replace('.geojson',''),
                                                           args.granule.replace('.h5',''))
    stats = [args.polygon[args.polygon.rfind('/')+1:].replace('simplified_', ''), args.granule]  # 流域名, granule 名
    stats += granule_stats          # 追加 4 项长度/光子统计
    with open(statsfname, 'w') as f: print('%s,%s,%.3f,%.3f,%i,%i' % tuple(stats), file=f)
except:
    print("could not write stats file")
    traceback.print_exc()

# ---------- 清理临时下载的 ATL03（处理完即删，节省磁盘）----------
if args.keep_granule:
    print('Keeping input granule:', input_filename)
elif os.path.isfile(input_filename):
    os.remove(input_filename)

print('\n-------------------------------------------------')
print(  '----------->   Python script done!   <-----------')
print(  '-------------------------------------------------\n')
