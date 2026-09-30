from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SUM=ROOT/'data'/'derived'/'summary_by_protocol_load.csv'
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
LABELS={'capde':'CAPDE-MAC','static':'Static reference','priority_csma':'Priority CSMA/CA','adt_mac':'ADT-MAC','mdp_hymac':'MDP-HYMAC','no_prediction':'No prediction','no_preemption':'No pre-emption'}


def plot_metric(df, metric, ylabel, filename, scale=1.0):
    fig,ax=plt.subplots(figsize=(7,4.6))
    for protocol,g in df.groupby('protocol'):
        g=g.sort_values('load')
        y=g[metric+'_mean']*scale
        yerr=g[metric+'_ci95_halfwidth']*scale
        ax.errorbar(g['load'],y,yerr=yerr,marker='o',capsize=3,label=LABELS.get(protocol,protocol))
    ax.set_xlabel('Offered-load multiplier')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/filename,dpi=300)
    plt.close(fig)


def main():
    df=pd.read_csv(SUM)
    plot_metric(df,'mean_delay_ms','Mean end-to-end delay (ms)','figure_mean_delay.png')
    plot_metric(df,'p95_delay_ms','95th-percentile delay (ms)','figure_p95_delay.png')
    plot_metric(df,'deadline_miss_ratio','Deadline-miss ratio','figure_deadline_miss.png')
    plot_metric(df,'packet_delivery_ratio','Packet delivery ratio','figure_pdr.png')
    plot_metric(df,'energy_per_delivered_mj','Energy per delivered packet (mJ)','figure_energy.png')
    print(f'Figures written to {OUT}')

if __name__=='__main__':
    main()
