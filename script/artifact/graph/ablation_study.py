import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from script.artifact.graph.accuracy_cal import *

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def render_figure8():
    print("Rendering figure 8...")
    os.chdir(os.path.expanduser('~/compass/'))
    d = "msmarco"

    ef_spec_x = []
    ef_spec_y = []

    ef_neighbor_x = []
    ef_neighbor_y = []

    compass_trip_acc = []
    compass_trip_latency = []

    nolazy_y = []
    nolazy_x = []
    nobatch_x = []

    # prepare mrr
    passage_file = "./data/dataset/msmarco_bert/passages/collection.tsv"
    query_file = "./data/dataset/msmarco_bert/passages/queries.dev.small.tsv"
    qrel_file = "./data/dataset/msmarco_bert/passages/qrels.dev.small.tsv"
    offset_to_qid, offset_to_pid, qrels = prepare_mrr_text(passage_file, query_file, qrel_file)

    # load results
    print("-> Loading ablation study results...")
    result_prefix = "./script/artifact/results/"

    efn = 24
    efspec = 1
    while efspec <= 16:
        # print("-> efspec: ", efspec)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        query_result = ivecs_read(f_accuracy)

        mrr = evaluate_mrr(query_result.tolist(), offset_to_qid, offset_to_pid, qrels)

        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        ef_spec_x.append(per_latency.mean())
        ef_spec_y.append(mrr)

        if efspec == 8:
            compass_trip_acc.append(mrr)
            compass_trip_latency.append(per_latency.mean())
            nolazy_y.append(mrr)

        efspec = efspec*2


    efspec = 8
    efn = 1
    while efn <= 256:
        # print("-> efn: ", efn)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        query_result = ivecs_read(f_accuracy)

        mrr = evaluate_mrr(query_result.tolist(), offset_to_qid, offset_to_pid, qrels)

        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        # if efn == 1:
        #     print("mrr: ", mrr)

        ef_neighbor_x.append(float(per_latency.mean()))
        ef_neighbor_y.append(mrr)
        efn = efn*2
    
    # lazy
    f_lazy_latency = result_prefix + f"ablation_latency_{d}_lazy.fvecs"
    lazy_latency = fvecs_read(f_lazy_latency)[0]
    nq = int(lazy_latency.shape[0] / 2)
    per_lazy_latency = lazy_latency[nq:]
    nolazy_x.append(per_lazy_latency.mean())

    # vanilla
    f_vanilla_latency = result_prefix + f"ablation_latency_{d}_vanilla.fvecs"
    vanilla_latency = fvecs_read(f_vanilla_latency)[0]
    nq = int(vanilla_latency.shape[0] / 2)
    per_vanilla_latency = vanilla_latency[nq:]
    nobatch_x.append(per_vanilla_latency.mean())

    print("-> Rendering...")
        
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': "Times New Roman"})

    # Create subplots for the broken axis
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 2.6), gridspec_kw={'width_ratios': [10, 2]})

    # ef_neighbor_x = [0.748, 0.752, 0.778, 0.837, 1.026, 1.402, 1.881, 3.732, 9.065]
    # ef_neighbor_y = [0.2252928435, 0.2763285692, 0.3086350457, 0.3315426161, 0.3465779779, 0.3581467458, 0.3603532201, 0.361916246, 0.3620322236]

    # ef_spec_y = [0.3603624869, 0.3597544572, 0.3571423455, 0.3551173989, 0.3350648679]
    # ef_spec_x = [4.522, 2.552, 1.651, 1.27, 1.077]

    # compass_trip_acc = [0.355]
    # compass_trip_latency = [1.251]

    ax1.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
    ax1.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

    ax1.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
    ax1.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

    ax1.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn=[1, 256]", marker='>', linestyle='-', zorder=2, markersize = 5)
    ax1.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec=[1, 16]" , zorder=2, markersize = 5)

    ax1.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", label="Compass", marker="*", s=200 , zorder=3)

    ax1.scatter(nolazy_x, nolazy_y, color="#0072B2", label="w/o Lazy Eviction", marker="*", s=200 , zorder=3)

    # ax1.set_title('MSMARCO', fontsize=14)
    ax1.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
    ax1.set_ylabel('MRR@10')
    # ax1.set_xscale('log')
    ax1.set_xticks([ 0, 2, 4, 6, 8])
    ax1.set_xlim(0, 10)
    ax1.set_yticks([ 0.20, 0.23, 0.26, 0.29, 0.32, 0.35])
    ax1.set_ylim(0.16, 0.38)
    # ax1.legend()
    ax1.grid(True, linestyle='--')
    ax1.spines['right'].set_color('gray')


    ax2.scatter(nobatch_x, nolazy_y, color="#009E73", label="vanilla Ring ORAM", marker="*", s=200 , zorder=3)
    ax2.set_xlim(104, 106)
    ax2.set_xticks([104, 105, 106])
    ax2.grid(True, linestyle='--')
    ax2.set_ylim(0.16, 0.38)
    ax2.spines['left'].set_color('gray')
    ax2.tick_params(axis='y', left=False)

    # Add diagonal lines to indicate the break
    d = 0.02  # size of diagonal line
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
    ax1.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d*2, +d*2), (-d, +d), **kwargs)
    ax2.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

    left  = 0.13  # the left side of the subplots of the figure
    right = 0.95   # the right side of the subplots of the figure
    bottom = 0.18  # the bottom of the subplots of the figure
    top = 0.98      # the top of the subplots of the figure
    wspace = 0.05  # the amount of width reserved for blank space between subplots
    hspace = 0.04   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(bottom=bottom, right=right, left=left, top=top, wspace=wspace, hspace=hspace)

    fig.legend(loc='lower right', bbox_to_anchor=(0.8, 0.18))

    os.makedirs('./eval_fig', exist_ok=True)
    plt.savefig('./eval_fig/figure8.pdf') 

if __name__ == "__main__":
    # render_figure8()
    render_figure8_trip()