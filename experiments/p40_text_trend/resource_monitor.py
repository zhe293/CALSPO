"""Streaming whole-card telemetry with explicit phase accounting and scoped cleanup."""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import psutil

def assert_free_gpus(selected):
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    mapping={}
    for line in query.splitlines():
        idx,uid,memory=[x.strip() for x in line.split(',')]
        if int(idx) in selected:
            mapping[uid]=int(idx)
            if float(memory)>100:
                raise RuntimeError(f'GPU {idx} already has {memory} MiB allocated; recheck ownership')
    if len(mapping)!=len(selected):
        raise RuntimeError('Requested GPU not found')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(line.split(',')[0].strip() in mapping for line in apps.splitlines()):
        raise RuntimeError('A selected GPU has an existing compute process')

def gpu_rows(selected):
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
                                  '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in query.splitlines():
        idx,util,mem,power=[x.strip() for x in line.split(',')]
        if int(idx) in selected:
            rows.append({'gpu':int(idx),'gpu_util_pct':float(util),'memory_mib':float(mem),'power_w':float(power)})
    if len(rows)!=len(selected):
        raise RuntimeError('Missing GPU telemetry')
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpus',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--log',type=Path,required=True)
    ap.add_argument('--interval',type=float,default=5)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.interval<=0:
        ap.error('Command and positive interval required')
    selected={int(x) for x in args.gpus.split(',')}
    gpu_rows(selected)  # telemetry must work before the child starts
    root=args.output.parent
    root.mkdir(parents=True,exist_ok=True)
    started=time.time()
    summary={'return_code':None,'wall_time_s':0,'gpu_hours':0,'gpus':{},'phases':{},'monitor_error':None,
             'scope':'whole selected GPUs; sampled peaks; CPU/RSS process tree is partial for Ray'}
    previous={}
    cpu_cache={}
    cpu_previous={}
    last_cpu_time=time.monotonic()
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    child=None
    owned={}
    def cleanup():
        if child is None:
            return
        # Only the exact child process group and descendants observed under it.
        for proc in list(owned.values()):
            try:
                if proc.is_running():
                    proc.terminate()
            except psutil.Error:
                pass
        if child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        _,alive=psutil.wait_procs(list(owned.values()),timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
    with args.log.open('w',encoding='utf-8',buffering=1) as log, args.output.open('w',newline='',encoding='utf-8') as output:
        writer=None
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                now=time.time()
                mono=time.monotonic()
                try:
                    descendants=[psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]
                except psutil.Error:
                    descendants=[]
                rss=0
                cpu_delta=0.0
                for proc in descendants:
                    try:
                        owned[(proc.pid,proc.create_time())]=proc
                        rss+=proc.memory_info().rss
                        usage=proc.cpu_times()
                        key=(proc.pid,proc.create_time())
                        value=usage.user+usage.system
                        if key in cpu_previous:
                            cpu_delta+=max(0,value-cpu_previous[key])
                        cpu_previous[key]=value
                    except psutil.Error:
                        pass
                phase='initialization'
                phase_path=root/'phase.json'
                if phase_path.exists():
                    phase=json.loads(phase_path.read_text())['phase']
                try:
                    telemetry_rows = gpu_rows(selected)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                    summary['telemetry_errors'] = summary.get('telemetry_errors', 0) + 1
                    print(f'[RESOURCE_SAMPLE_WARNING] {exc}', flush=True)
                    telemetry_rows = []
                    previous.clear()  # Do not fabricate energy across an unobserved gap.
                for row in telemetry_rows:
                    sample={'time':now,'elapsed_s':now-started,'phase':phase,
                            'partial_process_rss_gib':rss/1024**3,
                            'partial_cpu_pct':100*cpu_delta/max(1e-6,mono-last_cpu_time),**row}
                    if writer is None:
                        writer=csv.DictWriter(output,fieldnames=sample.keys())
                        writer.writeheader()
                    writer.writerow(sample)
                    stats=summary['gpus'].setdefault(str(row['gpu']),{'peak_memory_mib':0,'util_sum':0,'samples':0,'energy_wh':0})
                    stats['peak_memory_mib']=max(stats['peak_memory_mib'],row['memory_mib'])
                    stats['util_sum']+=row['gpu_util_pct']
                    stats['samples']+=1
                    if row['gpu'] in previous:
                        old=previous[row['gpu']]
                        dt=now-old['time']
                        energy=(old['power_w']+row['power_w'])/2*dt/3600
                        stats['energy_wh']+=energy
                        # Attribution refined after completion from exact phase event timestamps.
                    previous[row['gpu']]=sample
                output.flush()
                last_cpu_time=mono
                if child.poll() is not None:
                    summary['return_code']=child.returncode
                    break
                if stop.wait(args.interval):
                    raise InterruptedError('Monitor interrupted')
        except BaseException as exc:
            summary['monitor_error']=str(exc)
            cleanup()
            summary['return_code']=child.returncode if child.returncode else 1
        finally:
            cleanup()
    summary['wall_time_s']=time.time()-started
    summary['gpu_hours']=len(selected)*summary['wall_time_s']/3600
    summary['total_gpu_energy_wh']=sum(v['energy_wh'] for v in summary['gpus'].values())
    for stats in summary['gpus'].values():
        stats['mean_util_pct']=stats.pop('util_sum')/stats['samples']
    # Split each telemetry interval at driver phase boundaries, interpolating power.
    events=[{'time':started,'phase':'initialization'}]
    event_file=root/'phases.jsonl'
    if event_file.exists():
        events += [json.loads(line) for line in event_file.read_text().splitlines() if line]
    events.append({'time':time.time(),'phase':'end'})
    events.sort(key=lambda e:e['time'])
    with args.output.open() as f:
        samples=list(csv.DictReader(f))
    for start,end in zip(events,events[1:]):
        name=start['phase']
        val=summary['phases'].setdefault(name,{'wall_time_s':0,'energy_wh':0,'gpu_hours':0})
        dt=max(0,end['time']-start['time'])
        val['wall_time_s']+=dt
        val['gpu_hours']+=dt*len(selected)/3600
        for gpu in selected:
            rows=[r for r in samples if int(r['gpu'])==gpu]
            for left,right in zip(rows,rows[1:]):
                t0,t1=float(left['time']),float(right['time'])
                lo,hi=max(t0,start['time']),min(t1,end['time'])
                if hi<=lo or t1<=t0:
                    continue
                p0,p1=float(left['power_w']),float(right['power_w'])
                a=p0+(p1-p0)*(lo-t0)/(t1-t0)
                b=p0+(p1-p0)*(hi-t0)/(t1-t0)
                val['energy_wh']+=(a+b)/2*(hi-lo)/3600
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    raise SystemExit(summary['return_code'] or (1 if summary['monitor_error'] else 0))

if __name__=='__main__':
    main()
