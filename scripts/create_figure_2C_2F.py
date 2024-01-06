

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.rcdefaults()
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

if __name__ == "__main__":
    data_dir = '/Users/eghbalhosseini/Dropbox (MIT)/Desktop/NOL_final_submission/data/'
    benchmark='Pereira2018-encoding'
    trained_models=['mistral-gpt2-untrained','mistral-gpt2-step-00001','mistral-gpt2-step-00010','mistral-gpt2-step-00100','mistral-gpt2-step-01000','mistral-gpt2-step-10000']
    gpt2_model='gpt2'
    gpt2_untrained_model='gpt2-untrained'
    # load the train model performance
    score_data=[]
    scores_mean=[]
    scores_std=[]
    for model_id in trained_models:
        model_path=Path(data_dir,f'benchmark={benchmark},model={model_id},subsample=None.pkl')
        x=pd.read_pickle(model_path)['data']
        score_data.append(x.values)
        scores_mean.append(x.values[:,0])
        scores_std.append(x.values[:,1])
    # load the gpt2 model performance from Schrimpf et al. 2021
    model_path=Path(data_dir,f'benchmark={benchmark},model={gpt2_model},subsample=None.pkl')
    gpt2_model_score=pd.read_pickle(model_path)['data']

    # untrained model from Schrimpf et al. 2021
    model_path = Path(data_dir, f'benchmark={benchmark},model={gpt2_untrained_model},subsample=None.pkl')
    untrained_model_score = pd.read_pickle(model_path)['data']

    #%% find the layer with maximum score
    max_score_id=np.argmax(gpt2_model_score.values[:,0])
    layer_name = gpt2_model_score['layer'][max_score_id]
    scr_layer = [x[max_score_id] for x in scores_mean]
    scr_layer_std = [x[max_score_id] for x in scores_std]
    scr_gpt2 = gpt2_model_score.sel(layer=(gpt2_model_score.layer==layer_name))
    scr_untrained = untrained_model_score.sel(layer=(untrained_model_score.layer==layer_name))

    #%% plot the scores for training
    cmap_all=matplotlib.colormaps.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    chkpoints_label = ['0%','0.1%\n~10M', '1%\n~100M', '10%\n~1B', '100%\n~10B', '10*\n100%','Schrimpf\n(2021)']
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio= 8/11
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .7, .35, .25*pap_ratio))
    x_coords = [0.001, 0.01, 0.1, 1, 10,100]
    ylims = (-.12, 1.1)
    for idx, scr in enumerate(scr_layer):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10,markeredgecolor='k',markeredgewidth=1, zorder=2)
        ax.errorbar(x_coords[idx], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
    ax.set_xscale('log')
    ax.plot(x_coords[1:], scr_layer[1:], color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    # plot gpt2 score
    ax.plot(8e2, scr_gpt2.values[0][0], color=(.3,.3,.3,1), linewidth=2, marker='o', markersize=10, zorder=2)
    ax.errorbar(8e2, scr_gpt2.values[0][0], yerr=scr_gpt2.values[0][1], color='k', zorder=1)
    # plot untrained score
    ax.plot(0.0008, scr_untrained.values[0][0], color=all_col[0, :], linewidth=2, marker='o',markeredgecolor='w',markersize=10, zorder=2)
    ax.errorbar(0.0008, scr_untrained.values[0][0], yerr=scr_untrained.values[0][1],color='k', zorder=1)
    ax.set_xticks(np.concatenate([major_ticks,[8e2]]))
    ax.xaxis.grid(True, which="major", ls="-", color='0.9', zorder=0)
    ax.yaxis.grid(False)
    ax.set_xticklabels(chkpoints_label, rotation=0)
    ax.set_ylim(ylims)
    ax.set_xlim((0.0005, 1.5e3))
    ax.set_axisbelow(True)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    #%% plot scores per layer during training
    ax = plt.axes((.05, .35, .9, .25*pap_ratio))
    scores_mean_np = np.stack(scores_mean).transpose()
    scores_std_np = np.stack(scores_std).transpose()
    for idx, scr in enumerate(scores_mean_np):
        r3 = np.arange(len(scr))
        std_v=scores_std_np[idx,:]
        r3 = .95 * r3 / len(r3)
        r3= r3-np.mean(r3)
        for idy ,sc in enumerate(scr):
            ax.plot(r3[idy]+idx, sc, color=all_col[idy, :], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,  zorder=5)
            ax.errorbar(r3[idy]+idx, sc, yerr=std_v[idy], color='k', zorder=3)
        ax.plot(r3[1:] + idx, scr[1:], color=(.5,.5,.5), linewidth=1,zorder=4)
    ax.axhline(y=0, color='k', linestyle='-',zorder=2)
    ax.set_xlim((0 - .5, len(gpt2_model_score) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_mean_np)))
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylim(ylims)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} - layerwise')
    fig.show()