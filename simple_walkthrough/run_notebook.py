"""Execute this trusted local notebook without Jupyter dependencies; retain its outputs.

This runs the same code cells in order in one Python process. It is not a Jupyter
kernel/server. An optional notebook path may be supplied; execute only trusted code.
"""
import argparse
import base64
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import time
import traceback


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('notebook',nargs='?',type=Path,default=Path(__file__).with_name('QTabGAN_step_by_step.ipynb'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    path=args.notebook.resolve();output=(args.output or path).resolve()
    notebook=json.loads(path.read_text());namespace={'__name__':'__main__','NOTEBOOK_PATH':str(path)}
    count=0;failed=False;started=time.monotonic()
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        count+=1;cell['execution_count']=count;cell['outputs']=[]
        stdout=io.StringIO();stderr=io.StringIO()
        def flush():
            for name,buffer in [('stdout',stdout),('stderr',stderr)]:
                value=buffer.getvalue()
                if value:cell['outputs'].append({'output_type':'stream','name':name,'text':value.splitlines(True)})
                buffer.seek(0);buffer.truncate(0)
        def emit(data):
            flush();cell['outputs'].append({'output_type':'display_data','data':data,'metadata':{}})
        namespace['_emit_display']=emit
        print(f'Executing cell {count}...',flush=True)
        try:
            with redirect_stdout(stdout),redirect_stderr(stderr):
                exec(compile(''.join(cell['source']),f'{path.name}:cell_{count}','exec'),namespace)
        except Exception as exc:
            flush();cell['outputs'].append({'output_type':'error','ename':type(exc).__name__,'evalue':str(exc),
                                           'traceback':traceback.format_exc().splitlines()})
            print(traceback.format_exc(),flush=True);failed=True;break
        flush()
    # Clear stale outputs of cells that did not execute after an error.
    if failed:
        for cell in notebook['cells']:
            if cell['cell_type']=='code' and cell.get('execution_count',0) and cell['execution_count']>count:
                cell['execution_count']=None;cell['outputs']=[]
    notebook['metadata']['walkthrough_execution']={'engine':'Sequential Python cell execution (not a Jupyter kernel)',
        'successful':not failed,'elapsed_seconds':round(time.monotonic()-started,2)}
    temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(notebook,indent=1)+'\n');temporary.replace(output)
    print(f'Saved {output}; success={not failed}',flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
