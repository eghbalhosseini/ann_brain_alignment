
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
    benchmark = 'Pereira2018-encoding'
    trained_models = ['mistral-gpt2-untrained', 'mistral-gpt2-step-00001', 'mistral-gpt2-step-00010',
                      'mistral-gpt2-step-00100', 'mistral-gpt2-step-01000', 'mistral-gpt2-step-10000']
    gpt2_model = 'gpt2'
    gpt2_untrained_model = 'gpt2-untrained'

    preplex_benchmark = 'wikitext-103-raw-v1-test'
    trained_models_perplexity = [
                                    53099.7070,  # untrained
                                    21383.7090,  # 0.1%
                                    1439.5457,  # 1.0%
                                    75.1746,  # 10.0%
                                    43.0601,  # 100%
                                    35.9569  # 10x100%
                                ]
    gpt2_untrained_model_perplexity=56145.7422
    # load the train model performance
    score_data = []
    scores_mean = []
    scores_std = []
    for model_id in trained_models:
        model_path = Path(data_dir, f'benchmark={benchmark},model={model_id},subsample=None.pkl')
        x = pd.read_pickle(model_path)['data']
        score_data.append(x.values)
        scores_mean.append(x.values[:, 0])
        scores_std.append(x.values[:, 1])
    # load the gpt2 model performance from Schrimpf et al. 2021
    model_path = Path(data_dir, f'benchmark={benchmark},model={gpt2_model},subsample=None.pkl')
    gpt2_model_score = pd.read_pickle(model_path)['data']

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
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    ylims = (-.12, 1.1)
    xlims =(15,7e4)
    pap_ratio= 8/11
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .7, .45, .35*pap_ratio))
    ax.set_xscale('log')
    ax.plot(trained_models_perplexity[1:], scr_layer[1:], zorder=2, color=(0, 0, .5))
    for idx in range(len(trained_models_perplexity)):
        ax.plot(trained_models_perplexity[idx], scr_layer[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(trained_models_perplexity[idx], scr_layer[idx], yerr=scr_layer_std[idx], linewidth=2,
                    color=all_col[idx, :], marker=None, markersize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.set_axisbelow(True)
    ax.plot(gpt2_untrained_model_perplexity, scr_untrained.values[0][0], color=all_col[0, :], linewidth=2, marker='o', markeredgecolor='w',
            markersize=10, label=f'Unt.HF', zorder=2)
    ax.errorbar(gpt2_untrained_model_perplexity, scr_untrained.values[0][0], yerr=scr_untrained.values[0][1], color='k', zorder=1)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.invert_xaxis()
    ax.xaxis.grid(True, which="major", ls="-", color='0.9', zorder=0)
    ax.yaxis.grid(False)
    ax.set_ylim(ylims)
    ax.set_axisbelow(True)
    ax.set_title(f'benchmark {benchmark}')
    #%% plot scores per layer during training
    fig.show()