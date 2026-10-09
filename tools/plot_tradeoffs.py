"""Render resource/performance comparisons exclusively from matrix.json."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,default=Path('results/matrix/matrix.json'))
    args=ap.parse_args()
    d=json.loads(args.matrix.read_text())
    if d['problems'] or d['source_overrides']:raise ValueError('Refusing unverified/development matrix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'openvla':'#3567a7','vla_adapter':'#c36b27','fastwam':'#21866b'}
    names={'openvla':'OpenVLA','vla_adapter':'VLA-Adapter Pro','fastwam':'FastWAM (first-frame)'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for model,color in colors.items():
        rows=[r for r in d['configs'] if r['model']==model and r['method'] in ('fp16','rtn')
              and r['macro_success_rate'] is not None and r['resource_bench']['status']=='complete']
        if len(rows)!=2:continue
        rows.sort(key=lambda r:r['method']!='fp16')
        for ax,key in zip(axes,('memory','latency')):
            xs=[r['memory']['nvml_peak_bytes']/2**30 if key=='memory' else r['latency']['per_env_step_s']*1000 for r in rows]
            ys=[r['macro_success_rate']*100 for r in rows]
            ax.plot(xs,ys,color=color,linewidth=1.5,alpha=.7)
            for x,y,r,marker in zip(xs,ys,rows,('o','s')):
                ax.scatter(x,y,s=65,color=color,marker=marker)
            ax.plot([],[],color=color,label=names[model])
    axes[0].set_xlabel('Steady-state process peak (GiB; lower is better)')
    axes[1].set_xlabel('Amortized inference cost per action step (ms; log scale)')
    axes[1].set_xscale('log')
    for ax in axes:
        ax.set_ylabel('LIBERO macro success rate (%)');ax.set_ylim(69,101)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(loc='lower left',fontsize=8,frameon=False)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],marker='o',linestyle='none',color='#555555',label='Original precision'),
                           Line2D([],[],marker='s',linestyle='none',color='#555555',label='RTN')],
                   loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('Task 1.4: original precision vs RTN — measured on NVIDIA L40',fontsize=12)
    for ext in ('png','svg'):
        fig.savefig(args.matrix.parent/f'tradeoffs.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
