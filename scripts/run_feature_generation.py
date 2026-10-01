"""Run two ESM-C shards and pocket generation with process status reporting."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esmc-python',required=True)
    parser.add_argument('--pocket-python',required=True)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--existing-gpu1-pid',type=int)
    parser.add_argument('--existing-gpu0-pid',type=int)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    logs=root/'data/features/logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    jobs={};processes={}
    def start(name,command):
        with (logs/f'{name}.log').open('w') as output:
            process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                                     stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=process;jobs[name]=dict(pid=process.pid,command=command,state='running')
    for shard in (0,1):
        name=f'esmc_gpu{shard}'
        existing_pid = args.existing_gpu0_pid if shard==0 else args.existing_gpu1_pid
        if existing_pid:
            jobs[name]=dict(pid=existing_pid,state='running');continue
        start(name,[args.esmc_python,'scripts/generate_esmc_cache.py','--targets','data/features/targets.json',
              '--output-dir','data/features/esmc6b','--model-dir',args.model_dir,'--device',f'cuda:{shard}',
              '--shard',str(shard),'--num-shards','2'])
    start('pockets',[args.pocket_python,'scripts/generate_pocket_graphs.py','--targets','data/features/targets.json',
          '--output-dir','data/features/pockets','--embeddings','data/features/esmc6b','--workers',str(args.workers)])
    def write_status(state):
        payload=dict(state=state,supervisor_pid=os.getpid(),jobs=jobs,
                     cache_files=len(list((root/'data/features/esmc6b').glob('*.pt'))),
                     graph_files=len(list((root/'data/features/pockets/graphs').glob('*.pt'))))
        temp=logs/'status.json.tmp';temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(logs/'status.json')
    while True:
        failed=False
        for name,job in jobs.items():
            if job['state']!='running':continue
            if name in processes:
                code=processes[name].poll()
            else:
                proc=Path(f"/proc/{job['pid']}/stat")
                running=proc.exists() and proc.read_text().split(') ')[1][0]!='Z'
                if running:code=None
                else:code=0 if 'ESM-C shard complete' in (logs/f'{name}.log').read_text() else 1
            if code is not None:
                job.update(state='complete' if code==0 else 'failed',exit_code=code)
            failed=failed or job['state']=='failed'
        if failed:
            for name,job in jobs.items():
                if job['state']=='running':
                    try:os.killpg(job['pid'],15)
                    except ProcessLookupError:pass
                    job['state']='stopped'
            write_status('failed');return 1
        if all(j['state']=='complete' for j in jobs.values()):
            write_status('complete');return 0
        write_status('running');time.sleep(10)


if __name__=='__main__':
    sys.exit(main())
