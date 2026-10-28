"""Vector normalized-margin distribution plot from independently reloaded models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'revision_artifacts/checkpoint_reevaluation'
PAPER=ROOT.parent/'LDL-Lipschitz-ResNet-LMI'


def main():
    audit=json.loads((DATA/'image_audit.json').read_text())
    assert audit['complete'] and len(audit['rows'])==18
    radius=np.linspace(0,1,201)
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8.1,3.0),layout='constrained')
    source={'radius':radius}
    for alg,name,color in [('ldlt-resnet','LDL residual','#0072B2'),
                           ('sandwich','Sandwich','#009E73'),('sdp','SLL','#CC79A7')]:
        values={}
        for loss in ['ce','margin']:
            curves=[]
            for seed in [42,43,44]:
                data=np.load(DATA/f'cifar10-{alg}-{loss}-{seed}.npz')
                assert len(data['radius'])==10000
                curves.append(100*(data['radius'][:,None]>radius[None,:]).mean(0))
            values[loss]=np.array(curves)
            source[alg+'_'+loss]=values[loss]
        axes[0].plot(radius,values['margin'].mean(0),color=color,label=name)
        difference=values['margin']-values['ce']
        mean,sd=difference.mean(0),difference.std(0,ddof=1)
        axes[1].plot(radius,mean,color=color,label=name)
        axes[1].fill_between(radius,mean-sd,mean+sd,color=color,alpha=.12,linewidth=0)
    axes[0].set(title='Normalized-margin survival: margin training',ylabel='Test examples with certified radius > r (%)')
    axes[1].set(title='Margin minus CE: paired by training seed',ylabel='Certified accuracy difference (pp)')
    axes[1].axhline(0,color='gray',lw=.7)
    for ax in axes:
        ax.set_xlabel('Radius r in raw pixel coordinates');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(PAPER/'Figures/revision/cifar_margin_distributions.pdf')
    np.savez_compressed(DATA/'cifar_margin_distribution_curves.npz',**source)


if __name__=='__main__':main()
