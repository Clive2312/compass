import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from script.artifact.graph.accuracy_cal import *

# def render_figure8_trip():
#     print("Rendering figure 8...")
#     os.chdir(os.path.expanduser('~/compass/'))
    
#     d = "trip"

#     ef_spec_x = []
#     ef_spec_y = []

#     ef_neighbor_x = []
#     ef_neighbor_y = []

#     compass_trip_acc = []
#     compass_trip_latency = []

#     nolazy_y = []
#     nolazy_x = []
#     nobatch_x = []

#     # prepare mrr
#     passage_file = "./data/dataset/trip_distilbert/benchmark_tsv/documents/docs.tsv"
#     query_file = "./data/dataset/trip_distilbert/benchmark_tsv/topics/topics.head.val.tsv"
#     qrel_file = "./data/dataset/trip_distilbert/benchmark_tsv/qrels/qrels.dctr.head.val.tsv"
#     offset_to_qid, offset_to_pid, qrels = prepare_mrr_text(passage_file, query_file, qrel_file)

#     # load results
#     print("-> Loading ablation study results...")
#     result_prefix = "./script/artifact/results/"

#     efn = 24
#     efspec = 1
#     while efspec <= 16:
#         # print("-> efspec: ", efspec)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = evaluate_mrr(query_result.tolist(), offset_to_qid, offset_to_pid, qrels)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         ef_spec_x.append(per_latency.mean())
#         ef_spec_y.append(mrr)

#         efspec = efspec*2


#     efspec = 6
#     efn = 1
#     while efn <= 256:
#         # print("-> efn: ", efn)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = evaluate_mrr(query_result.tolist(), offset_to_qid, offset_to_pid, qrels)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         # if efn == 1:
#         #     print("mrr: ", mrr)

#         ef_neighbor_x.append(float(per_latency.mean()))
#         ef_neighbor_y.append(mrr)
#         efn = efn*2
    
#     # lazy
#     f_lazy_latency = result_prefix + f"ablation_latency_{d}_lazy.fvecs"
#     lazy_latency = fvecs_read(f_lazy_latency)[0]
#     nq = int(lazy_latency.shape[0] / 2)
#     per_lazy_latency = lazy_latency[nq:]
#     nolazy_x.append(per_lazy_latency.mean())

#     # vanilla
#     f_vanilla_latency = result_prefix + f"ablation_latency_{d}_vanilla.fvecs"
#     vanilla_latency = fvecs_read(f_vanilla_latency)[0]
#     nq = int(vanilla_latency.shape[0] / 2)
#     per_vanilla_latency = vanilla_latency[nq:]
#     nobatch_x.append(per_vanilla_latency.mean())


#     f_latency  = result_prefix + f"latency_slow_trip.fvecs"
#     f_accuracy  = result_prefix + f"accuracy_trip.ivecs"

#     latency = fvecs_read(f_latency)[0]
#     query_result = ivecs_read(f_accuracy)

#     mrr = evaluate_mrr(query_result.tolist(), offset_to_qid, offset_to_pid, qrels)

#     nq = int(latency.shape[0] / 2)
#     per_latency = latency[0:nq]

#     compass_trip_acc.append(mrr)
#     compass_trip_latency.append(per_latency.mean())
#     nolazy_y.append(mrr)

#     print("-> Rendering...")
        
#     plt.rcParams.update({'font.size': 14})
#     plt.rcParams.update({'font.family': "Times New Roman"})

#     # Create subplots for the broken axis
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 2.6), gridspec_kw={'width_ratios': [10, 2]})

#     # ef_neighbor_x = [0.748, 0.752, 0.778, 0.837, 1.026, 1.402, 1.881, 3.732, 9.065]
#     # ef_neighbor_y = [0.2252928435, 0.2763285692, 0.3086350457, 0.3315426161, 0.3465779779, 0.3581467458, 0.3603532201, 0.361916246, 0.3620322236]

#     # ef_spec_y = [0.3603624869, 0.3597544572, 0.3571423455, 0.3551173989, 0.3350648679]
#     # ef_spec_x = [4.522, 2.552, 1.651, 1.27, 1.077]

#     # compass_trip_acc = [0.355]
#     # compass_trip_latency = [1.251]

#     ax1.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
#     ax1.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

#     ax1.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
#     ax1.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

#     ax1.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn=[1, 256]", marker='>', linestyle='-', zorder=2, markersize = 5)
#     ax1.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec=[1, 16]" , zorder=2, markersize = 5)

#     ax1.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", label="Compass", marker="*", s=100 , zorder=3)

#     ax1.scatter(nolazy_x, nolazy_y, color="#0072B2", label="w/o Lazy Eviction", marker="*", s=100 , zorder=3)

#     # ax1.set_title('MSMARCO', fontsize=14)
#     ax1.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
#     ax1.set_ylabel('MRR@10')
#     # ax1.set_xscale('log')
#     ax1.set_xticks([ 0, 2, 4, 6])
#     ax1.set_xlim(0, 6)
#     ax1.set_yticks([ 0.17, 0.20, 0.23, 0.26, 0.29])
#     ax1.set_ylim(0.16, 0.32)
#     # ax1.legend()
#     ax1.grid(True, linestyle='--')
#     ax1.spines['right'].set_color('gray')

#     print(nobatch_x)
#     ax2.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
#     ax2.set_xlim(75, 77)
#     ax2.set_xticks([75, 76, 77])
#     ax2.grid(True, linestyle='--')
#     ax1.set_ylim(0.16, 0.32)
#     ax2.spines['left'].set_color('gray')
#     ax2.tick_params(axis='y', left=False)

#     # Add diagonal lines to indicate the break
#     d = 0.02  # size of diagonal line
#     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

#     kwargs.update(transform=ax2.transAxes)
#     ax2.plot((-d*2, +d*2), (-d, +d), **kwargs)
#     ax2.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

#     left  = 0.13  # the left side of the subplots of the figure
#     right = 0.95   # the right side of the subplots of the figure
#     bottom = 0.18  # the bottom of the subplots of the figure
#     top = 0.98      # the top of the subplots of the figure
#     wspace = 0.05  # the amount of width reserved for blank space between subplots
#     hspace = 0.04   # the amount of height reserved for white space between subplots
#     plt.subplots_adjust(bottom=bottom, right=right, left=left, top=top, wspace=wspace, hspace=hspace)

#     fig.legend(loc='lower right', bbox_to_anchor=(0.8, 0.18))

#     os.makedirs('./eval_fig', exist_ok=True)
#     plt.savefig('./eval_fig/figure8_trip.pdf') 

# def render_figure8_sift():
#     print("Rendering figure 8...")
#     os.chdir(os.path.expanduser('~/compass/'))
#     d = "sift"

#     ef_spec_x = []
#     ef_spec_y = []

#     ef_neighbor_x = []
#     ef_neighbor_y = []

#     compass_trip_acc = []
#     compass_trip_latency = []

#     nolazy_y = []
#     nolazy_x = []
#     nobatch_x = []

#     # load results
#     print("-> Loading ablation study results...")
#     result_prefix = "./script/artifact/results/"

#     gt = "./data/dataset/sift/gt.ivecs"

#     efn = 12
#     efspec = 1
#     while efspec <= 8:
#         # print("-> efspec: ", efspec)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = compute_mrr_non_text(gt,f_accuracy)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         ef_spec_x.append(per_latency.mean())
#         ef_spec_y.append(mrr)

#         efspec = efspec*2


#     efspec = 4
#     efn = 1
#     while efn <= 128:
#         # print("-> efn: ", efn)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = compute_mrr_non_text(gt,f_accuracy)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         # if efn == 1:
#         #     print("mrr: ", mrr)

#         ef_neighbor_x.append(float(per_latency.mean()))
#         ef_neighbor_y.append(mrr)
#         efn = efn*2
    
#     # lazy
#     f_lazy_latency = result_prefix + f"ablation_latency_{d}_lazy.fvecs"
#     lazy_latency = fvecs_read(f_lazy_latency)[0]
#     nq = int(lazy_latency.shape[0] / 2)
#     per_lazy_latency = lazy_latency[nq:]
#     nolazy_x.append(per_lazy_latency.mean())

#     # vanilla
#     f_vanilla_latency = result_prefix + f"ablation_latency_{d}_vanilla.fvecs"
#     vanilla_latency = fvecs_read(f_vanilla_latency)[0]
#     nq = int(vanilla_latency.shape[0] / 2)
#     per_vanilla_latency = vanilla_latency[nq:]
#     nobatch_x.append(per_vanilla_latency.mean())


#     f_latency  = result_prefix + f"latency_slow_{d}.fvecs"
#     f_accuracy  = result_prefix + f"accuracy_{d}.ivecs"

#     latency = fvecs_read(f_latency)[0]
#     query_result = ivecs_read(f_accuracy)

#     mrr = compute_mrr_non_text(gt,f_accuracy)

#     nq = int(latency.shape[0] / 2)
#     per_latency = latency[0:nq]

#     compass_trip_acc.append(mrr)
#     compass_trip_latency.append(per_latency.mean())
#     nolazy_y.append(mrr)

#     print("-> Rendering...")
        
#     plt.rcParams.update({'font.size': 14})
#     plt.rcParams.update({'font.family': "Times New Roman"})

#     # Create subplots for the broken axis
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 2.6), gridspec_kw={'width_ratios': [10, 2]})

#     # ef_neighbor_x = [0.748, 0.752, 0.778, 0.837, 1.026, 1.402, 1.881, 3.732, 9.065]
#     # ef_neighbor_y = [0.2252928435, 0.2763285692, 0.3086350457, 0.3315426161, 0.3465779779, 0.3581467458, 0.3603532201, 0.361916246, 0.3620322236]

#     # ef_spec_y = [0.3603624869, 0.3597544572, 0.3571423455, 0.3551173989, 0.3350648679]
#     # ef_spec_x = [4.522, 2.552, 1.651, 1.27, 1.077]

#     # compass_trip_acc = [0.355]
#     # compass_trip_latency = [1.251]

#     ax1.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
#     ax1.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

#     ax1.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
#     ax1.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

#     ax1.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn=[1, 256]", marker='>', linestyle='-', zorder=2, markersize = 5)
#     ax1.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec=[1, 16]" , zorder=2, markersize = 5)

#     ax1.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", label="Compass", marker="*", s=100 , zorder=3)

#     ax1.scatter(nolazy_x, nolazy_y, color="#0072B2", label="w/o Lazy Eviction", marker="*", s=100 , zorder=3)

#     # ax1.set_title('MSMARCO', fontsize=14)
#     ax1.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
#     ax1.set_ylabel('MRR@10')
#     # ax1.set_xscale('log')
#     ax1.set_xticks([ 0, 2])
#     ax1.set_xlim(0, 2)
#     # ax1.set_yticks([ 0.17, 0.20, 0.23, 0.26, 0.29])
#     ax1.set_ylim(0.6, 1.0)
#     # ax1.legend()
#     ax1.grid(True, linestyle='--')
#     ax1.spines['right'].set_color('gray')

#     print(nobatch_x)
#     ax2.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
#     ax2.set_xlim(21, 22)
#     ax2.set_xticks([21, 22])
#     ax2.grid(True, linestyle='--')
#     ax2.set_ylim(0.6, 1.0)
#     ax2.spines['left'].set_color('gray')
#     ax2.tick_params(axis='y', left=False)

#     # Add diagonal lines to indicate the break
#     d = 0.02  # size of diagonal line
#     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

#     kwargs.update(transform=ax2.transAxes)
#     ax2.plot((-d*2, +d*2), (-d, +d), **kwargs)
#     ax2.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

#     left  = 0.13  # the left side of the subplots of the figure
#     right = 0.95   # the right side of the subplots of the figure
#     bottom = 0.18  # the bottom of the subplots of the figure
#     top = 0.98      # the top of the subplots of the figure
#     wspace = 0.05  # the amount of width reserved for blank space between subplots
#     hspace = 0.04   # the amount of height reserved for white space between subplots
#     plt.subplots_adjust(bottom=bottom, right=right, left=left, top=top, wspace=wspace, hspace=hspace)

#     fig.legend(loc='lower right', bbox_to_anchor=(0.8, 0.18))

#     os.makedirs('./eval_fig', exist_ok=True)
#     plt.savefig('./eval_fig/figure8_sift.pdf') 

# def render_figure8_laion():
#     print("Rendering figure 8...")
#     os.chdir(os.path.expanduser('~/compass/'))
#     d = "laion"

#     ef_spec_x = []
#     ef_spec_y = []

#     ef_neighbor_x = []
#     ef_neighbor_y = []

#     compass_trip_acc = []
#     compass_trip_latency = []

#     nolazy_y = []
#     nolazy_x = []
#     nobatch_x = []

#     # load results
#     print("-> Loading ablation study results...")
#     result_prefix = "./script/artifact/results/"

#     gt = "./data/dataset/laion1m/100k/gt.ivecs"

#     efn = 12
#     efspec = 1
#     while efspec <= 8:
#         # print("-> efspec: ", efspec)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = compute_mrr_non_text(gt,f_accuracy)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         ef_spec_x.append(per_latency.mean())
#         ef_spec_y.append(mrr)

#         efspec = efspec*2

#     print(ef_spec_x)
#     print(ef_spec_y)
#     efspec = 2
#     efn = 1
#     while efn <= 128:
#         # print("-> efn: ", efn)
#         f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
#         f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

#         # mean perceived latency
#         latency = fvecs_read(f_latency)[0]
#         query_result = ivecs_read(f_accuracy)

#         mrr = compute_mrr_non_text(gt,f_accuracy)

#         nq = int(latency.shape[0] / 2)
#         per_latency = latency[0:nq]

#         # if efn == 1:
#         #     print("mrr: ", mrr)

#         ef_neighbor_x.append(float(per_latency.mean()))
#         ef_neighbor_y.append(mrr)
#         efn = efn*2

#     print(ef_neighbor_x)
#     print(ef_neighbor_y)
    
#     # lazy
#     f_lazy_latency = result_prefix + f"ablation_latency_{d}_lazy.fvecs"
#     lazy_latency = fvecs_read(f_lazy_latency)[0]
#     nq = int(lazy_latency.shape[0] / 2)
#     per_lazy_latency = lazy_latency[nq:]
#     nolazy_x.append(per_lazy_latency.mean())

#     # vanilla
#     f_vanilla_latency = result_prefix + f"ablation_latency_{d}_vanilla.fvecs"
#     vanilla_latency = fvecs_read(f_vanilla_latency)[0]
#     nq = int(vanilla_latency.shape[0] / 2)
#     per_vanilla_latency = vanilla_latency[nq:]
#     nobatch_x.append(per_vanilla_latency.mean())


#     f_latency  = result_prefix + f"latency_slow_{d}.fvecs"
#     f_accuracy  = result_prefix + f"accuracy_{d}.ivecs"

#     latency = fvecs_read(f_latency)[0]
#     query_result = ivecs_read(f_accuracy)

#     mrr = compute_mrr_non_text(gt,f_accuracy)

#     nq = int(latency.shape[0] / 2)
#     per_latency = latency[0:nq]

#     compass_trip_acc.append(mrr)
#     compass_trip_latency.append(per_latency.mean())
#     nolazy_y.append(mrr)

#     print("-> Rendering...")
        
#     plt.rcParams.update({'font.size': 14})
#     plt.rcParams.update({'font.family': "Times New Roman"})

#     # Create subplots for the broken axis
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 2.6), gridspec_kw={'width_ratios': [10, 2]})

#     # ef_neighbor_x = [0.748, 0.752, 0.778, 0.837, 1.026, 1.402, 1.881, 3.732, 9.065]
#     # ef_neighbor_y = [0.2252928435, 0.2763285692, 0.3086350457, 0.3315426161, 0.3465779779, 0.3581467458, 0.3603532201, 0.361916246, 0.3620322236]

#     # ef_spec_y = [0.3603624869, 0.3597544572, 0.3571423455, 0.3551173989, 0.3350648679]
#     # ef_spec_x = [4.522, 2.552, 1.651, 1.27, 1.077]

#     # compass_trip_acc = [0.355]
#     # compass_trip_latency = [1.251]

#     ax1.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
#     ax1.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

#     ax1.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
#     ax1.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

#     ax1.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn=[1, 256]", marker='>', linestyle='-', zorder=2, markersize = 5)
#     ax1.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec=[1, 16]" , zorder=2, markersize = 5)

#     ax1.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", label="Compass", marker="*", s=100 , zorder=3)

#     ax1.scatter(nolazy_x, nolazy_y, color="#0072B2", label="w/o Lazy Eviction", marker="*", s=100 , zorder=3)

#     # ax1.set_title('MSMARCO', fontsize=14)
#     ax1.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
#     ax1.set_ylabel('MRR@10')
#     # ax1.set_xscale('log')
#     ax1.set_xticks([ 0, 1, 2])
#     ax1.set_xlim(0, 3)
#     # ax1.set_yticks([ 0.17, 0.20, 0.23, 0.26, 0.29])
#     ax1.set_ylim(0.4, 1.05)
#     # ax1.legend()
#     ax1.grid(True, linestyle='--')
#     ax1.spines['right'].set_color('gray')

#     print(nobatch_x)
#     ax2.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
#     ax2.set_xlim(11, 13)
#     ax2.set_xticks([11, 12, 13])
#     ax2.grid(True, linestyle='--')
#     ax2.set_ylim(0.4, 1.05)
#     ax2.spines['left'].set_color('gray')
#     ax2.tick_params(axis='y', left=False)

#     # Add diagonal lines to indicate the break
#     d = 0.02  # size of diagonal line
#     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
#     ax1.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

#     kwargs.update(transform=ax2.transAxes)
#     ax2.plot((-d*2, +d*2), (-d, +d), **kwargs)
#     ax2.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

#     left  = 0.13  # the left side of the subplots of the figure
#     right = 0.95   # the right side of the subplots of the figure
#     bottom = 0.18  # the bottom of the subplots of the figure
#     top = 0.98      # the top of the subplots of the figure
#     wspace = 0.05  # the amount of width reserved for blank space between subplots
#     hspace = 0.04   # the amount of height reserved for white space between subplots
#     plt.subplots_adjust(bottom=bottom, right=right, left=left, top=top, wspace=wspace, hspace=hspace)

#     fig.legend(loc='lower right', bbox_to_anchor=(0.8, 0.18))

#     os.makedirs('./eval_fig', exist_ok=True)
#     plt.savefig('./eval_fig/figure8_laion.pdf') 


def render_figure8_all():

    print("Rendering figure 8...")
    os.chdir(os.path.expanduser('~/compass/'))

    acc_cal = AccCal()

    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': "Times New Roman"})

    fig = plt.figure(figsize=(12, 2.8))

    intra_gap = 0.05
    inner_gap = 1.4

    # Define GridSpec with precise control over subplot widths and spaces
    gs = GridSpec(1, 15, figure=fig, width_ratios=[
        6, intra_gap,  # ax8 and gap after ax8
        1,  inner_gap,  # ax7 and gap after ax7
        6, intra_gap,  # ax6 and gap after ax6
        1,  inner_gap,  # ax5 and gap after ax5
        6, intra_gap,  # ax3 and gap after ax3
        1,  inner_gap,  # ax4 and gap after ax4
        9, intra_gap,  # ax1 and gap after ax1
        1         # ax2 (no gap after last subplot)
    ])

    # Assign subplots explicitly
    ax7 = fig.add_subplot(gs[0, 0])
    ax8 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[0, 4])
    ax6 = fig.add_subplot(gs[0, 6])
    ax3 = fig.add_subplot(gs[0, 8])
    ax4 = fig.add_subplot(gs[0, 10])
    ax1 = fig.add_subplot(gs[0, 12])
    ax2 = fig.add_subplot(gs[0, 14])

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
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

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
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

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

    # ax1.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
    # ax1.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

    # ax1.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
    # ax1.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

    ax1.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn:[1,256]", marker='>', linestyle='-', zorder=2, markersize = 5)
    ax1.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec:[1,16]" , zorder=2, markersize = 5)

    ax1.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", marker="*", s=100 , zorder=3)

    ax1.scatter(nolazy_x, nolazy_y, color="#0072B2", marker="*", s=100 , zorder=3)

    ax1.set_title('MS MARCO', fontsize=14)
    ax1.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
    # ax1.set_ylabel('MRR@10')
    # ax1.set_xscale('log')
    ax1.set_xticks([ 0, 2, 4, 6, 8])
    ax1.set_xlim(0, 9)
    ax1.set_yticks([0.16, 0.22, 0.28, 0.34, 0.40])
    ax1.set_ylim(0.16, 0.40)
    ax1.legend(loc='lower right', edgecolor='none', handletextpad=0.3, labelspacing=0.2, borderpad=0.25, borderaxespad=0.2, handlelength=1.5)
    ax1.grid(True, linestyle='--')
    ax1.spines['right'].set_color('gray')


    ax2.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
    ax2.set_xlim(104, 106)
    ax2.set_xticks([105])
    ax2.grid(True, linestyle='--')
    ax2.set_ylim(0.16, 0.40)
    ax2.set_yticks([0.16, 0.22, 0.28, 0.34, 0.40])
    ax2.spines['left'].set_color('gray')
    ax2.tick_params(axis='y', labelleft=False, labelright=False)

    # Add diagonal lines to indicate the break
    d = 0.02  # size of diagonal line
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
    ax1.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d*2, +d*2), (-d, +d), **kwargs)
    ax2.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)


    # --------------------------------------  

    d = "trip"

    ef_spec_x = []
    ef_spec_y = []

    ef_neighbor_x = []
    ef_neighbor_y = []

    compass_trip_acc = []
    compass_trip_latency = []

    nolazy_y = []
    nolazy_x = []
    nobatch_x = []

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
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

        ef_spec_x.append(per_latency.mean())
        ef_spec_y.append(mrr)

        efspec = efspec*2


    efspec = 6
    efn = 1
    while efn <= 256:
        # print("-> efn: ", efn)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

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


    f_latency  = result_prefix + f"latency_slow_trip.fvecs"
    f_accuracy  = result_prefix + f"accuracy_trip.ivecs"

    latency = fvecs_read(f_latency)[0]
    mrr = acc_cal.compute_mrr(d, f_accuracy)

    nq = int(latency.shape[0] / 2)
    per_latency = latency[0:nq]

    compass_trip_acc.append(mrr)
    compass_trip_latency.append(per_latency.mean())
    nolazy_y.append(mrr)

    # ax3.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
    # ax3.text(ef_neighbor_x[-1] - 1.05, ef_neighbor_y[-1] + 0.004, "efn=256")

    # ax3.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.025, "efspec=1")
    # ax3.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

    ax3.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn:[1,256]", marker='>', linestyle='-', zorder=2, markersize = 5)
    ax3.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec:[1,16]" , zorder=2, markersize = 5)

    ax3.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", marker="*", s=100 , zorder=3)

    ax3.scatter(nolazy_x, nolazy_y, color="#0072B2", marker="*", s=100 , zorder=3)

    ax3.set_title('TripClick', fontsize=14)
    ax3.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
    # ax3.set_ylabel('MRR@10')
    # ax3.set_xscale('log')
    ax3.set_xticks([ 0, 2, 4, 6])
    ax3.set_xlim(0, 6)
    ax3.set_yticks([0.15, 0.19, 0.23, 0.27, 0.31])
    ax3.set_ylim(0.15, 0.31)
    ax3.legend(loc='lower right', edgecolor='none', handletextpad=0.3, labelspacing=0.2, borderpad=0.25, borderaxespad=0.2, handlelength=1.5)
    ax3.grid(True, linestyle='--')
    ax3.spines['right'].set_color('gray')

    print(nobatch_x)
    ax4.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
    ax4.set_xlim(75, 77)
    ax4.set_xticks([76])
    ax4.grid(True, linestyle='--')
    ax4.set_yticks([0.15, 0.19, 0.23, 0.27, 0.31])
    ax4.set_ylim(0.15, 0.31)
    ax4.spines['left'].set_color('gray')
    ax4.tick_params(axis='y', labelleft=False, labelright=False)

    # Add diagonal lines to indicate the break
    d = 0.02  # size of diagonal line
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
    ax3.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax4.transAxes)
    ax4.plot((-d*2, +d*2), (-d, +d), **kwargs)
    ax4.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

    # --------------------------------------  

    d = "sift"

    ef_spec_x = []
    ef_spec_y = []

    ef_neighbor_x = []
    ef_neighbor_y = []

    compass_trip_acc = []
    compass_trip_latency = []

    nolazy_y = []
    nolazy_x = []
    nobatch_x = []

    gt = "./data/dataset/sift/gt.ivecs"

    efn = 12
    efspec = 1
    while efspec <= 8:
        # print("-> efspec: ", efspec)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]
        mrr = acc_cal.compute_mrr(d, f_accuracy)

        ef_spec_x.append(per_latency.mean())
        ef_spec_y.append(mrr)

        efspec = efspec*2


    efspec = 4
    efn = 1
    while efn <= 128:
        # print("-> efn: ", efn)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

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


    f_latency  = result_prefix + f"latency_slow_{d}.fvecs"
    f_accuracy  = result_prefix + f"accuracy_{d}.ivecs"

    latency = fvecs_read(f_latency)[0]
    mrr = acc_cal.compute_mrr(d, f_accuracy)

    nq = int(latency.shape[0] / 2)
    per_latency = latency[0:nq]

    compass_trip_acc.append(mrr)
    compass_trip_latency.append(per_latency.mean())
    nolazy_y.append(mrr)

    # ax5.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] + 0.002, "efn=1")
    # ax5.text(ef_neighbor_x[-1] - 0.3, ef_neighbor_y[-1] + 0.02, "efn=128")

    # ax5.text(ef_spec_x[0] - 0.3, ef_spec_y[0] - 0.03, "efspec=1")
    # ax5.text(ef_spec_x[-1] + 0.25, ef_spec_y[-1] - 0.01, "efspec=16")

    ax5.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn:[1,128]", marker='>', linestyle='-', zorder=2, markersize = 5)
    ax5.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec:[1,8]" , zorder=2, markersize = 5)

    ax5.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", marker="*", s=100 , zorder=3)

    ax5.scatter(nolazy_x, nolazy_y, color="#0072B2", marker="*", s=100 , zorder=3)

    ax5.set_title('SIFT1M', fontsize=14)
    ax5.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
    # ax5.set_ylabel('MRR@10')
    # ax5.set_xscale('log')
    ax5.set_xticks([ 0, 1, 2, 3])
    ax5.set_xlim(0, 3)
    ax5.set_yticks([0.6, 0.70, 0.80, 0.90, 1])
    ax5.set_ylim(0.6, 1.0)
    ax5.legend(loc='lower right', edgecolor='none', handletextpad=0.3, labelspacing=0.2, borderpad=0.25, borderaxespad=0.2, handlelength=1.5)
    ax5.grid(True, linestyle='--')
    ax5.spines['right'].set_color('gray')

    print(nobatch_x)
    ax6.scatter(nobatch_x, nolazy_y, color="grey", label="vanilla Ring ORAM", marker="*", s=100 , zorder=3)
    ax6.set_xlim(21, 23)
    ax6.set_xticks([22])
    ax6.grid(True, linestyle='--')
    ax6.set_yticks([0.6, 0.70, 0.80, 0.90, 1])
    ax6.set_ylim(0.6, 1.0)
    ax6.spines['left'].set_color('gray')
    ax6.tick_params(axis='y', labelleft=False, labelright=False)

    # Add diagonal lines to indicate the break
    d = 0.02  # size of diagonal line
    kwargs = dict(transform=ax5.transAxes, color='k', clip_on=False)
    ax5.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
    ax5.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax6.transAxes)
    ax6.plot((-d*2, +d*2), (-d, +d), **kwargs)
    ax6.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

    # --------------------------------------  

    d = "laion"

    ef_spec_x = []
    ef_spec_y = []

    ef_neighbor_x = []
    ef_neighbor_y = []

    compass_trip_acc = []
    compass_trip_latency = []

    nolazy_y = []
    nolazy_x = []
    nobatch_x = []

    # load results
    print("-> Loading ablation study results...")
    result_prefix = "./script/artifact/results/"

    gt = "./data/dataset/laion1m/100k/gt.ivecs"

    efn = 12
    efspec = 1
    while efspec <= 8:
        # print("-> efspec: ", efspec)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

        ef_spec_x.append(per_latency.mean())
        ef_spec_y.append(mrr)

        efspec = efspec*2

    print(ef_spec_x)
    print(ef_spec_y)
    efspec = 2
    efn = 1
    while efn <= 128:
        # print("-> efn: ", efn)
        f_latency  = result_prefix + f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        f_accuracy  = result_prefix + f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"

        # mean perceived latency
        latency = fvecs_read(f_latency)[0]
        nq = int(latency.shape[0] / 2)
        per_latency = latency[0:nq]

        mrr = acc_cal.compute_mrr(d, f_accuracy)

        # if efn == 1:
        #     print("mrr: ", mrr)

        ef_neighbor_x.append(float(per_latency.mean()))
        ef_neighbor_y.append(mrr)
        efn = efn*2

    print(ef_neighbor_x)
    print(ef_neighbor_y)
    
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


    f_latency  = result_prefix + f"latency_slow_{d}.fvecs"
    f_accuracy  = result_prefix + f"accuracy_{d}.ivecs"

    latency = fvecs_read(f_latency)[0]
    mrr = acc_cal.compute_mrr(d, f_accuracy)

    nq = int(latency.shape[0] / 2)
    per_latency = latency[0:nq]

    compass_trip_acc.append(mrr)
    compass_trip_latency.append(per_latency.mean())
    nolazy_y.append(mrr)

    # ax7.text(ef_neighbor_x[0] + 0.15, ef_neighbor_y[0] -  0.005, "efn=1")
    # ax7.text(ef_neighbor_x[-1] - 0.03, ef_neighbor_y[-1] - 0.07, "efn=128")

    # ax7.text(ef_spec_x[0] - 0.08, ef_spec_y[0] - 0.07, "efspec=1")
    # ax7.text(ef_spec_x[-1] + 0.15, ef_spec_y[-1] - 0.01, "efspec=16")

    legend_efn, = ax7.plot(ef_neighbor_x, ef_neighbor_y, color="#fd8282", label="efn:[1,128]", marker='>', linestyle='-', zorder=2, markersize = 5)
    legend_efspec, = ax7.plot(ef_spec_x, ef_spec_y, marker='<', linestyle='-', color="#019e73", label="efspec:[1,8]" , zorder=2, markersize = 5)

    legend_compass = ax7.scatter(compass_trip_latency, compass_trip_acc, color="#e6a002", label="Compass" ,marker="*", s=100 , zorder=3)

    legend_compass_lazy = ax7.scatter(nolazy_x, nolazy_y, color="#0072B2", label="Compass w/o Lazy Eviction" , marker="*", s=100 , zorder=3)

    ax7.set_title('LAION', fontsize=14)
    ax7.set_xlabel('Latency (s)', fontsize=14, labelpad = 0)
    ax7.set_ylabel('MRR@10')
    # ax7.set_xscale('log')
    ax7.set_xticks([ 0, 1, 2, 3])
    ax7.set_xlim(0, 3)
    ax7.set_yticks([0.4, 0.55, 0.70, 0.85, 1])
    ax7.set_ylim(0.4, 1)
    ax7.legend([legend_efn, legend_efspec], [legend_efn.get_label(), legend_efspec.get_label()], edgecolor='none', loc='lower right', handletextpad=0.3, labelspacing=0.2, borderpad=0.25, borderaxespad=0.2, handlelength=1.5)
    ax7.grid(True, linestyle='--')
    ax7.spines['right'].set_color('gray')

    print(nobatch_x)
    legend_compass_vanilla = ax8.scatter(nobatch_x, nolazy_y, color="grey", label="Compass w/ vanilla Ring ORAM", marker="*", s=100 , zorder=3)
    ax8.set_xlim(11, 13)
    ax8.set_xticks([12])
    ax8.grid(True, linestyle='--')
    ax8.set_yticks([0.4, 0.55, 0.70, 0.85, 1])
    ax8.set_ylim(0.4, 1)
    ax8.spines['left'].set_color('gray')
    ax8.tick_params(axis='y', labelleft=False, labelright=False)

    # Add diagonal lines to indicate the break
    d = 0.02  # size of diagonal line
    kwargs = dict(transform=ax7.transAxes, color='k', clip_on=False)
    ax7.plot(((1 - d*0.4), 1 + d*0.4), (-d, +d), **kwargs)
    ax7.plot(((1 - d*0.4), 1 + d*0.4), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax8.transAxes)
    ax8.plot((-d*2, +d*2), (-d, +d), **kwargs)
    ax8.plot((-d*2, +d*2), (1 - d, 1 + d), **kwargs)

    global_handles = [legend_compass, legend_compass_lazy, legend_compass_vanilla]
    global_labels = [h.get_label() for h in global_handles]

    fig.legend(global_handles, global_labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=14)

    left  = 0.06  # the left side of the subplots of the figure
    right = 0.99   # the right side of the subplots of the figure
    bottom = 0.18  # the bottom of the subplots of the figure
    top = 0.82      # the top of the subplots of the figure
    wspace = 0.05  # the amount of width reserved for blank space between subplots
    hspace = 0.04   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(bottom=bottom, right=right, left=left, top=top, wspace=wspace, hspace=hspace)

    os.makedirs('./eval_fig', exist_ok=True)
    plt.savefig('./eval_fig/figure8_full_2.pdf')
