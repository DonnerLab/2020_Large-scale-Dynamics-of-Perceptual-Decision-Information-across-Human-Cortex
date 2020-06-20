"""
Make figures for manuscript.

Map of functions to figures:

Figure 1 -> figure1(...)
Figure 2 -> figure2_alt(...)
Figure 3 -> figure5(...)
Figure 4 -> figure7(...)
Figure 5 -> nr_figure5(...)

Figure S1 -> figureS1(...)
Figure S2 -> nr_figureS2A(...)
Figure S3 -> nr_figureS3(...)
Figure S4 -> nr_figureS4(...)
Figure S5 -> nr_figureS5(...)
Figure S6 -> nr_figureS6(...)
Figure S7 -> figureS3(...)
Figure S8 -> nr_figureS8(...)
Figure S9 -> Separate in matlab
Figure S10 -> nr_figureS10(...)
Figure S11 -> nr_figureS11(...)

"""

import pandas as pd
import matplotlib
from pylab import *
from conf_analysis.behavior import metadata
from conf_analysis.meg import decoding_plots as dp
from joblib import Memory
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_rel, linregress

memory = Memory(location=metadata.cachedir, verbose=-1)

import matplotlib.font_manager as font_manager
path = '/Users/nwilming/font_copies/Helvetica-01.ttf'
prop = font_manager.FontProperties(fname=path)
# Set font property dict
matplotlib.rcParams['font.family'] = 'Helvetica'

rc = {"font.size": 7.0, 'xtick.labelsize':7.0, 
    #'xlabel.font.size':7.0, 
    #'ylabel.font.size':7.0, 
    'ytick.labelsize':7.0,
    'legend.fontsize':7.0,
    "lines.linewidth": 1, 
    'font.family':prop.get_name()}
#label_size = 8
#mpl.rcParams['xtick.labelsize'] = label_size 

def_ig = slice(0.4, 1.1)


def array_to_data_source_file(figure_nr, panel, description, data, labels):
    import h5py
    filename = '/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/data_source_files/figure%s.hdf'%str(figure_nr)
    with h5py.File(filename, 'a') as F:        
        try:
            group = F[panel]            
        except KeyError:
            group = F.create_group(panel)
        
        if description in group:
            del group[description]
        dset = group.create_dataset(description, data=data)

        for name, values in labels.items():
            if (description+'_'+name) in group:
                del group[description+'_'+name]
            if (description+'_'+name+'_name') in group:
                del group[description+'_'+name+'_name']
        
            group[description+'_'+name] = values
            group[description+'_'+name+'_name'] = name
        

def table_to_data_source_file(figure_nr, panel, description, data):
    import h5py
    filename = '/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/data_source_files/figure%s.hdf'%str(figure_nr)
    with h5py.File(filename, 'a') as F:        
        try:
            group = F[panel]            
        except KeyError:
            group = F.create_group(panel)
        
        if description in group:
            del group[description]
        dset = group.create_dataset(description, data=data.values)
        if (description+'row_index') in group:
            del group[description+'row_index']
        if (description+'row_index_names') in group:
            del group[description+'row_index_names']
        if (description+'col_index') in group:
            del group[description+'col_index']
        if (description+'col_index_names') in group:
            del group[description+'col_index_names']
        try:
            group[description+'row_index'] = data.index.values
        except TypeError:
            group[description+'row_index'] = np.array([str(x) for x in data.index.values], dtype='S')
        group[description+'row_index_names'] = str(data.index.name)
        try:
            group[description+'col_index'] = data.columns.values
        except TypeError:
            group[description+'col_index'] = np.array([str(x) for x in data.columns.values], dtype='S')
        group[description+'col_index_names'] = str(data.columns.name)
        

def plotwerr(pivottable, *args, ax=None, ls="-", label=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    N = pivottable.shape[0]
    x = pivottable.columns.values
    mean = pivottable.mean(0).values
    std = pivottable.std(0).values
    sem = std / (N ** 0.5)
    ax.plot(x, mean, *args, label=label, **kwargs)
    if "alpha" in kwargs:
        del kwargs["alpha"]
    if "color" in kwargs:
        color = kwargs["color"]
        del kwargs["color"]
        ax.fill_between(
            x,
            mean + sem,
            mean - sem,
            facecolor=color,
            edgecolor="none",
            alpha=0.5,
            **kwargs
        )
    else:
        ax.fill_between(x, mean + sem, mean - sem, edgecolor="none", alpha=0.5, **kwargs)
    # for col in pivottable:
    #    sem = pivottable.loc[:, col].std() / pivottable.shape[0] ** 0.5
    #    m = pivottable.loc[:, col].mean()
    #    plot([col, col], [m - sem, m + sem], *args, **kwargs)


def draw_sig(
    ax, pivottable, y=0, color="k", fdr=False, lw=2, p=0.05, 
    conjunction=None, cluster_test=False, debug=False, **kwargs
):
    if not cluster_test:
        from scipy.stats import ttest_1samp
        p_sig = ttest_1samp(pivottable, 0)[1]
    else:
        p_sig = ct(pivottable.values)
    if fdr:
        from mne.stats import fdr_correction
        id_sig, p_sig = fdr_correction(p_sig)
        id_sig = list(id_sig)
    else:
        id_sig = list(p_sig < p)

    if debug:
        print('P-Values for test %s (fdr=%s, cluster=%s):'%(debug, fdr, cluster_test), p_sig)
    if conjunction is not None:
        p_con_sig = ttest_1samp(conjunction, 0)[1]
        id_con_sig = p_con_sig < p
        id_sig = list(np.array(id_sig) & id_con_sig)
    x = pivottable.columns.values
    d = np.where(np.diff([False] + id_sig + [False]))[0]
    dx = np.diff(x).astype(float)[0] / 10
    # xb = np.linspace(x.min()-dx, x.max()+dx, 5000)
    for low, high in zip(d[0:-1:2], d[1::2]):

        ax.plot([x[low] - dx, x[high - 1] + dx], [y, y], color=color, lw=lw, **kwargs)
    return p_sig


def _stream_palette():
    rois = [
        "vfcPrimary",
        "vfcEarly",
        "vfcV3ab",
        "vfcIPS01",
        "vfcIPS23",
        "JWG_aIPS",
        "vfcLO",
        "vfcTO",
        "vfcVO",
        "vfcPHC",
        "JWG_IPS_PCeS",
        "JWG_M1",
    ]
    return {
        roi: color
        for roi, color in zip(
            rois, sns.color_palette("viridis", n_colors=len(rois) + 1)
        )
    }


def figure0(gs=None):
    from matplotlib.patches import ConnectionPatch

    with mpl.rc_context(rc=rc):
        if gs is None:
            figure(figsize=(7.5, 4))
            gs = matplotlib.gridspec.GridSpec(
                2,
                6,
                height_ratios=[1, 1.5],
                width_ratios=[1, 0.25, 1, 1, 1, 1],
                hspace=0.1,
                wspace=0.00,
            )
        else:
            gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                2,
                6,
                gs,
                height_ratios=[1, 1.5],
                width_ratios=[1, 0.25, 1, 1, 1, 1],
                hspace=0.1,
                wspace=0.00,
            )
        ax = subplot(gs[1, :])

        def set(low, high, value=1, set_nan=False, x=None):
            if x is None:
                x = time
            out = x * 0
            out[(low <= x) & (x <= high)] = value
            if set_nan:
                idx = ~(out == value)
                diff = np.where(np.diff(idx))[0]
                idx[diff[0]] = False
                idx[diff[1] + 1] = False
                out[idx] = nan
            return out

        yticks = [
            (5, ""),  #'Reference\ncontrast'),
            #(4, "Delay"),
            (5, "Contrast"),
            (4, "Response (choice/confidence)"),
            #(2, "Feedback delay"),
            (3, "Audio feedback"),
        ]
        time = np.linspace(-1.25, 2.1, 5000)
        ref_time = np.linspace(-1.25, -0.1, 5000)
        stim_time = np.linspace(-0.1, 2.1, 5000)
        # Reference
        ax.plot(ref_time, 5 + set(-0.8, -0.4, 0.5, x=ref_time), "k", zorder=5)
        ax.text(-0.35, 5.35, "0.4s", va="center")
        # Reference delay
        #ax.plot(time, 4 + set(-0.4, -0.0, 0.5, set_nan=True), ":k", zorder=5)
        # ax.plot(time, 4 + set(-100, -200.0, 0.5, set_nan=False), "k", zorder=5)
        #ax.plot([time.min(), -0.4], [4, 4], "k", zorder=5)
        #ax.plot([0, time.max()], [4, 4], "k", zorder=5)
        #ax.text(0.05, 4.25, "1-1.5s", va="center")
        # Test stimulus
        cvals = array([0.71, 0.33, 0.53, 0.75, 0.59, 0.57, 0.55, 0.61, 0.45, 0.58])
        #colors = sns.color_palette(n_colors=10)
        
        norm=matplotlib.colors.Normalize(-5, 10)
        cm = matplotlib.cm.get_cmap('BuPu')
        colors = [cm(norm(10-i)) for i, c in enumerate(cvals)]
        for i in range(10):
            if i == 0:
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=True, x=stim_time),
                    color=colors[i],
                    zorder=10,
                )
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=False, x=stim_time),
                    "k",
                    zorder=5,
                )
            else:
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=True, x=stim_time),
                    color=colors[i],
                )
        ax.text(1.05, 5.35, "100ms/sample", va="center")
        # Response
        ax.plot(time, 4 + set(1.449, 1.45, 0.5, set_nan=True), "k", zorder=5)
        ax.plot([time.min(), time.max()], [4, 4], "k", zorder=5)
        #ax.plot([1.45, time.max()], [3, 3], "k", zorder=5)
        #Z2 = plt.imread('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/resp.png')
        #aspect = Z2.shape[1]/Z2.shape[0]
        #print('IMG Aspect:', aspect)
        #tt = 0.13
        #axins = ax.inset_axes([0.85, 0.42, tt*aspect, tt], zorder=-1)
        
        #axins.imshow(np.flipud(Z2), extent=[aspect,0, 1, 0], interpolation="nearest",
        #  origin="lower")
        #axins.set_xticks([])
        #axins.set_yticks([])
        #sns.despine(ax=axins, left=True, bottom=True)
        #ax.text(1.5, 4.25, "0.45s avg", va="center")
        # Feedback delay
        #ax.plot(time, 2 + set(1.45, 1.65, 0.5), ":k", zorder=5)
        # ax.plot(time, 2 + set(100.35, 100.55, 0.5), "k", zorder=5)
        #ax.text(1.7, 2.25, "0-1.5s", va="center")
        #ax.plot([time.min(), 1.45], [2, 2], "k", zorder=5)
        #ax.plot([1.65, time.max()], [2, 2], "k", zorder=5)
        # Feedback
        ax.plot(time, 3 + set(1.65, 1.65 + 0.25, 0.5), "k", zorder=5)
        ax.text(1.925, 3.35, "0.25s", va="center")

        ax.set_yticks([])  # i[0] for i in yticks])
        # ax.set_yticklabels([i[1] for i in yticks], va='bottom')
        for y, t in yticks:
            ax.text(-1.25, y + 0.35, t, verticalalignment="center")
        
        ax.set_xticks([])
        ax.tick_params(axis=u"both", which=u"both", length=0)
        ax.set_xlim(-1.25, 2.1)
        ax.set_ylim(2.8, 6.6)
        sns.despine(ax=ax, left=True, bottom=True)
        subax = subplot(gs[0, 0])

        height = 0.25
        pad = 0.1
        overlap = height / 2
        # subax = plt.gcf().add_axes([pad, 1-height-pad, height, height])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(0.5, ringwidth=8 / 4)[:, 400:-400]
        aspect = img.shape[0] / img.shape[1]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        plt.setp(subax.spines.values(), color="k", linewidth=2)
        xyA = (-0.8, 5.5)  # in axes coordinates
        xyB = (0, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        xyA = (-0.4, 5.5)  # in axes coordinates
        xyB = (1, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        subax = subplot(gs[0, 2])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[0], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[0], linewidth=2)

        xyA = (0, 5 + cvals[0])  # in axes coordinates
        xyB = (0, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        subax = subplot(gs[0, 3])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[1], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[1], linewidth=2)

        subax = subplot(gs[0, 4])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[2], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[2], linewidth=2)

        subax = subplot(gs[0, 5])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[3], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[3], linewidth=2)

        xyA = (0.4, 5 + cvals[3])  # in axes coordinates
        xyB = (1, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)
    return img


def invax(fig, gs):
    ax = fig.add_subplot(gs, zorder=-10)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    return ax


def add_letter(fig, gs, label, x=-0.05, y=1.1, new_axis=True):
    if new_axis:
        ax = invax(fig, gs)
    else:
        ax = gs
    ax.text(x, y, label, transform=ax.transAxes,
      fontsize=8, fontweight='bold', fontfamily='Helvetica', va='top', ha='right')
    return ax


def by_discrim(sd, abs=False):
    cvals = np.stack(sd.contrast_probe)
    threshold = (
        (cvals[sd.side == 1].mean(1).mean() - 0.5)
        - (cvals[sd.side == -1].mean(1).mean() - 0.5)
    ) / 2
    if not abs:
        edges = np.linspace(
            (0.5 + threshold) - 4 * threshold,
            (0.5 + threshold) + (2 * threshold),
            11,
        )
        centers = np.around(
            ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold))
            / threshold,
            2,
        )
        d = (
            sd.groupby([pd.cut(sd.mc.values, edges, labels=centers)])
            .mean()
            .loc[:, ["choice", "pconf", "mc", "correct"]]
            .reset_index()
        )
    else:
        edges = np.linspace(
            (0.5 + threshold) - 1 * threshold,
            (0.5 + threshold) + (2 * threshold),
            7,
        )
        centers = np.around(
            ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold))
            / threshold,
            2,
        )
        sd.loc[:, "mc"] = np.abs(sd.mc - 0.5) + 0.5
        d = (
            sd.groupby([pd.cut(sd.mc.values, edges, labels=centers)])
            .mean()
            .loc[:, ["choice", "pconf", "mc", "correct"]]
            .reset_index()
        )
    d.columns = ["threshold_units", "choice", "pconf", "mc", "accuracy"]
    k = d.threshold_units
    d.loc[:, "threshold_units"] = d.loc[:, "threshold_units"].astype(float) + 1
    return d


def figure1(data=None, slow=False):
    from conf_analysis.behavior import empirical, kernels
    from conf_analysis import behavior

    color_palette = behavior.parse(behavior.colors)
    if data is None:
        data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1

    fig=figure(figsize=(5.5, 7.5))
    gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[1, 0.6, 0.6], hspace=0.25)

    figure0(gs[0, 0])
    add_letter(fig, gs[0,0], 'A', x=-0.1, y=1.1)
    #sns.despine(ax=plt.gca(), left=True, bottom=True)
    
    with mpl.rc_context(rc=rc):
        #gs_new = matplotlib.gridspec.GridSpecFromSubplotSpec(
        #    2, 6, gs[1, 0], wspace=1.5, hspace=0.15
        #)
        # gs = matplotlib.gridspec.GridSpec(2, 6, wspace=1.5, hspace=0.5)

        
        ax = subplot(gs[1, :])
        #Add Panel 1B here!
        # This is the kernel panel
        #ax = subplot(gs[0, :])
        palette = {
            r"$E_{N}^{High}$": '#c64588', # color_palette["Secondary1"][0],
            r"$E_{N}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{High}$": '#c64588', #color_palette["Secondary1"][0],
        }

        kernel2descript = {
            r"$E_{N}^{High}$": 'high_conf_choice_weaker', 
            r"$E_{N}^{Low}$": 'low_conf_choice_weaker', 
            r"$E_{S}^{Low}$": 'low_conf_choice_stronger', 
            r"$E_{S}^{High}$": 'high_conf_choice_stronger'
        }
        k = empirical.get_confidence_kernels(data, contrast_mean=0.5)
        for kernel, kdata in k.groupby("Kernel"):
            kk = pd.pivot_table(
                index="snum", columns="time", values="contrast", data=kdata
            )
            table_to_data_source_file(1, "B", kernel2descript[kernel], kk)
            if 'E_{N}' in kernel:
                plotwerr(kk, '--', color=palette[kernel], lw=2, label="Low confidence")                
            else:
                plotwerr(kk, color=palette[kernel], lw=2, label="Low confidence")
                
            #print(kernel)
        #empirical.plot_kernel(k, palette, legend=False)
        plt.ylabel(r"$\Delta$ Contrast", fontsize=7)
        ax.annotate('Choice: test stronger, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, 0.03), xytext=(3.5, 0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)
        ax.annotate('Choice: test stronger, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(2.0, 0.0075), xytext=(4.5, 0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(3.7, -0.009), xytext=(6, -0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, -0.03), xytext=(3.5, -0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)

        for p in np.arange(10):
            ax.axvline(p, lw=0.5, color='gray', alpha=0.5, zorder=-1000)
        
        xticks(np.arange(10), np.arange(10) + 1, fontsize=7)
        
        ax.plot([0, 9], [0, 0], 'k')
        ax.set_ylim(-0.045, 0.045)
        xlim([-.5, 9.5])
        sns.despine(ax=ax)        

        add_letter(plt.gcf(), gs[1, :], 'B', x=-0.1)

        #---> Comment in for kernel:
        from conf_analysis.meg import cort_kernel as ck
        dk, confidence_kernel = ck.get_kernels()
        ax = subplot(gs[2, :])
        plotwerr(dk, color="k", label="Decision kernel", lw=2)
        table_to_data_source_file(1, "A", 'psychiphysical_kernel', dk)
        draw_sig(ax, dk, fdr=True, color="k", y=0.0)
        
        ylim([0.49 - 0.5, 0.64 - 0.5])
        ylabel("AUC-0.5", fontsize=7)
        xlabel("Contrast sample number", fontsize=7)
        legend('', frameon=False)
        yticks(np.array([0.5, 0.55, 0.6]) - 0.5)
        xticks(np.arange(10), np.arange(10) + 1)
        for p in np.arange(10):
            ax.axvline(p, lw=0.5, color='gray', alpha=0.5, zorder=-1000)
        xlim([-.5, 9.5])
        sns.despine(ax=gca(), bottom=False)

        add_letter(fig, gs[2,:], 'C', x=-0.1, y=1.1)
        #kernels.plot_decision_kernel_values(data)
        #ylabel("Contrast")
        #xlabel("Contrast sample")
        #legend(frameon=False)
        #add_letter(fig, gs[0,2:], 'C', x=-0.125)

        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/1_figure_1.pdf",
        #    bbox_inches="tight",
        #    dpi=1200,
        #)
        if slow:
            p = data.groupby(["snum"]).apply(
                lambda x: empirical.fit_choice_logistic(x, summary=False)
            )
            p = p.groupby("snum").mean()
            print(
                "Can predict choice above chance in %i/15 subjects. \nMean choice accuracy is %0.2f (%0.2f-%0.2f)"
                % ((sum(p > 0.5)), np.around(np.mean(p), 2), np.around(p.min(), 2), np.around(p.max(), 2))
            )

            p = data.groupby(["snum", "response"]).apply(
                lambda x: empirical.fit_conf_logistic(x, summary=False)
            )
            p = p.groupby("snum").mean()
            print(
                "Can predict confidence above chance in %i/15 subjects. \nMean confidence accuracy is %0.2f (%0.2f-%0.2f)"
                % ((sum(p > 0.5)), np.around(np.mean(p), 2), np.around(p.min(), 2), np.around(p.max(), 2))
            )
    return


def figureS1():
    from conf_analysis.behavior import empirical
    from conf_analysis import behavior
    data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1
    color_palette = behavior.parse(behavior.colors)
    fig=figure(figsize=(7.5, 2.5))
    with mpl.rc_context(rc=rc):
        gs = matplotlib.gridspec.GridSpec(
            1, 6, wspace=1.5, hspace=0.55
        )
            # This panel needs to go into supplements
        subplot(gs[0, :2])
        
        X = pd.pivot_table(index="snum", columns="pconf", values="correct", data=data)
        mean = X.mean(0)
        table_to_data_source_file('S1', 'A', 'correct_by_confidence', X)
        mean_all = X.mean().mean()
        for i in X.index.values:
            jitter = np.random.randn(1)*0.05
            plt.plot([0+jitter, 1+jitter], X.loc[i, :], '-', color="k",
                markerfacecolor='none', alpha=0.25, lw=1, zorder=-20)
            plt.plot([0+jitter, 1+jitter], X.loc[i, :], 'o', color="xkcd:medium gray",
                markerfacecolor='white', alpha=1)
            

        sem = 2 * X.std(0) / (15 ** 0.5)
        semall = 2 * X.mean(1).std() / (15 ** 0.5)        
        plot([-0.4], mean_all, 'o', color='gray')
        plot([-0.4, -0.4], [semall+mean_all, mean_all-semall], color='gray')
        plot([-0.2, .2], [mean[0], mean[0]], "-", lw=3, color=color_palette["Secondary2"][0], zorder=100)
        plot([.8, 1.2], [mean[1], mean[1]], "-", lw=3, color=color_palette["Secondary1"][0], zorder=100)
        #plot(
        #    [0, 0], [sem[0] + mean[0], mean[0] - sem[0]], color_palette["Secondary2"][0]
        #)
        #plot(
        #    [1, 1], [sem[1] + mean[1], mean[1] - sem[1]], color_palette["Secondary1"][0]
        #)
        xticks([-0.4, 0, 1], ["All\ntrials", r"Low", r"High"])
        ylabel("% Correct", fontsize=7)
        xlabel("Confidence", fontsize=7)
        xlim(-0.6, 1.3)
        from scipy.stats import ttest_rel, ttest_1samp

        print("T-Test for accuracy by confidence:", ttest_rel(X.loc[:, 0], X.loc[:, 1]))
        sns.despine(ax=gca())

        add_letter(plt.gcf(), gs[0, :2], 'A', x=-0.25)
        subplot(gs[0, 4:])

        dz = (
            data.groupby(["snum", "correct"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dz.loc[:, "Evidence discriminability"] = dz.threshold_units
        crct = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="pconf",
            data=dz.query("correct==1.0"),
        )
        err = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="pconf",
            data=dz.query("correct==0.0"),
        )
        from scipy.stats import linregress

        scrct, serr = [], []
        for s in range(1, 16):
            scrct.append(linregress(np.arange(5), crct.loc[s, :2.3])[0])
            serr.append(linregress(np.arange(5), err.loc[s, :2.3])[0])
        print(
            "Slopes for confidence vs. evidence (correct: mean, t, p, #>0):",
            np.mean(scrct),
            ttest_1samp(scrct, 0),
            sum(np.array(scrct) > 0),
        )
        print(
            "Slopes for confidence vs. evidence (error: mean, t, p, #<0):",
            np.mean(serr),
            ttest_1samp(serr, 0),
            sum(np.array(serr) < 0),
        )
        plotwerr(crct, color="g", lw=2, label="Correct")
        table_to_data_source_file('S1', 'C', 'Correct', crct)
        plotwerr(err, color="r", lw=2, label="Error")
        table_to_data_source_file('S1', 'C', 'Error', err)
        legend(frameon=False)

        xticks([1, 2], [r"t", r"2t"])
        xlabel("Evidence discriminability", fontsize=7)
        ylabel("Confidence", fontsize=7)
        yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        sns.despine(ax=gca())
        add_letter(plt.gcf(), gs[0, 4:], 'C', x=-0.25)

        subplot(gs[0, 2:4])
        dz = (
            data.groupby(["snum", "confidence"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dz.loc[:, "Evidence discriminability"] = dz.threshold_units
        dall = (
            data.groupby(["snum"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dall.loc[:, "Evidence discriminability"] = dall.threshold_units
        dall = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dall,
        )        
        plotwerr(
           dall,
            color="gray",
            alpha=1,
            lw=2,
            label="All",
        )
        table_to_data_source_file('S1', 'B', 'All', dall)

        high = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dz.query("pconf==1.0"),
        )
        et = high.loc[:, :2.2].columns.values

        plotwerr(
            high,
            color=color_palette["Secondary1"][0],
            alpha=1,
            lw=2,
            label="High confidence",
        )
        table_to_data_source_file('S1', 'B', 'High_confidence', high)
        low = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dz.query("pconf==0.0"),
        )
        plotwerr(
            low,
            color=color_palette["Secondary2"][0],
            alpha=1,
            lw=2,
            label="Low confidence",
        )
        table_to_data_source_file('S1', 'B', 'Low_confidence', low)
        hslope, lslope = [], []
        for s in range(1, 16):
            hslope.append(linregress(et, high.loc[s, :2.2])[0])
            lslope.append(linregress(et, low.loc[s, :2.2])[0])
        print(
            "Slopes for high confidence vs. evidence (mean, t, p, #>0):",
            np.mean(hslope),
            ttest_1samp(hslope, 0),
            sum(np.array(hslope) > 0),
        )
        print(
            "Slopes for low confidence vs. evidence (error: mean, t, p, #<0):",
            np.mean(lslope),
            ttest_1samp(lslope, 0),
            sum(np.array(lslope) < 0),
        )
        print("High vs low slope:", ttest_rel(hslope, lslope))
        legend(frameon=False)
        xticks([1, 2], [r"t", r"2t"], fontsize=7)
        xlabel("Evidence discriminability", fontsize=7)
        ylabel("% Correct", fontsize=7)
        yticks([0.5, 0.75, 1], [50, 75, 100], fontsize=7)
        sns.despine(ax=gca())
        # tight_layout()
        add_letter(plt.gcf(), gs[0, 2:4], 'B', x=-0.3)
        """
        ax = subplot(gs[1, :4])
        #Add Panel 1B here!
        # This is the kernel panel
        #ax = subplot(gs[0, :])
        palette = {
            r"$E_{N}^{High}$": '#c64588', # color_palette["Secondary1"][0],
            r"$E_{N}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{High}$": '#c64588', #color_palette["Secondary1"][0],
        }

        
        k = empirical.get_confidence_kernels(data, contrast_mean=0.5)
        for kernel, kdata in k.groupby("Kernel"):
            kk = pd.pivot_table(
                index="snum", columns="time", values="contrast", data=kdata
            )
            if 'E_{N}' in kernel:
                plotwerr(kk, '--', color=palette[kernel], lw=2, label="Low confidence")
            else:
                plotwerr(kk, color=palette[kernel], lw=2, label="Low confidence")
            print(kernel)
        #empirical.plot_kernel(k, palette, legend=False)
        plt.ylabel(r"$\Delta$ Contrast", fontsize=7)
        ax.annotate('Choice: test stronger, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, 0.03), xytext=(3.5, 0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)
        ax.annotate('Choice: test stronger, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(2.0, 0.0075), xytext=(4.5, 0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(3.7, -0.009), xytext=(6, -0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, -0.03), xytext=(3.5, -0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)

        for p in np.arange(10):
            ax.axvline(p, lw=0.5, color='gray', alpha=0.5, zorder=-1000)
        #plt.text(
        #    -0.2,
        #    0.003,
        #    "ref.    test",
        #    rotation=90,
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        xlabel("Contrast sample number", fontsize=7)
        # legend(frameon=False)
        xticks(np.arange(10), np.arange(10) + 1, fontsize=7)
        #ax.axhline(color="k", lw=1)
        ax.plot([0, 9], [0, 0], 'k')
        ax.set_ylim(-0.045, 0.045)
        xlim([-.5, 9.5])
        sns.despine(ax=ax)        

        add_letter(plt.gcf(), gs[1, :], 'D', x=-0.1)
        """
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/S1_figure_S1.pdf",
            bbox_inches="tight",
            dpi=1200,
        )


def figure2(df=None, stats=False):
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_stats_20190516.pickle",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190516.hdf"
        )
    palette = _stream_palette()
    with mpl.rc_context(rc=rc):
        figure(figsize=(7.5, 7.5 / 2))
        from conf_analysis.meg import srtfr

        # gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        fig = srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["all"],
            flip_cbar=True,
            # gs=gs[0, 0],
            stats=stats,
            title_palette=palette,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-50, 50), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        # [left, bottom, width, height]
        cax = fig.add_axes([0.74, 0.2, 0.1, 0.015])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            orientation="horizontal",
            ticks=[-50, 0, 50],
            drawedges=False,
            label="% Power\nchange",
        )
        cb.outline.set_visible(False)
        sns.despine(ax=cax)

        view_one = dict(azimuth=-40, elevation=100, distance=350)
        view_two = dict(azimuth=-145, elevation=70, distance=350)
        img = _get_palette(palette, views=[view_two, view_one])
        iax = fig.add_axes([0.09, 0.75, 0.25, 0.25])
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])
        sns.despine(ax=iax, left=True, bottom=True)

    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/2_figure_2.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return img, df, stats


def figure2_alt(df=None, stats=False, dcd=None, aspect='auto'):
    """
    Plot TFRs underneath each other.
    """
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_stats_20190516.pickle",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190516.hdf"
        )

    if dcd is None:
        dcd = dp.get_decoding_data()
    palette = _stream_palette()

    with mpl.rc_context(rc=rc):
        fig = figure(figsize=(7.5, 7.5))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(4, 1, height_ratios=[1,1, 1,0.1])

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["all"],
            flip_cbar=False,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-50, 50), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.685, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-50, 0, 50],
            drawedges=False,
            orientation="horizontal",            
        )
        cb.set_label(label="% Power change", fontsize=7)
        
        cb.outline.set_visible(False)
        view_one = dict(azimuth=-40, elevation=100, distance=350)
        view_two = dict(azimuth=-145, elevation=70, distance=350)
        img = _get_palette(palette, views=[view_two, view_one])
        iax = fig.add_axes([0.09, 0.8, 0.19, 0.19])
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])                
        lax = add_letter(plt.gcf(), gs[0,0], 'A', x=-0.08, y=1.25)        
        sns.despine(ax=lax, left=True, bottom=True)
        sns.despine(ax=iax, left=True, bottom=True)
        

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["stimulus"],
            flip_cbar=False,
            gs=gs[1, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.45, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-25, 0, 25],
            drawedges=False,
            orientation="horizontal",            
        )
        cb.set_label(label="% Power change", fontsize=7)
        
        lax = add_letter(plt.gcf(), gs[1,0], 'B', x=-0.08, y=1.1)
        sns.despine(ax=lax, left=True, bottom=True)
        
        
        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "k"},#, "CONF_unsigned": "Greens", "CONF_signed": "Blues"},
            {
                "Pair": dcd.test_roc_auc.Pair,
                #"Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[2, 0],
            title_palette=palette,
        )

        plotter.plot(aspect="auto")
        ### Data source files:
        dcd_data = (dcd
            .test_roc_auc
            .Pair
            .query('signal=="MIDC_split"')
            .loc[:, 
                ['JWG_IPS_PCeS', 'JWG_M1', 'JWG_aIPS', 'vfcEarly', 
                 'vfcFEF', 'vfcIPS01', 'vfcIPS23', 'vfcLO', 'vfcPHC', 
                 'vfcPrimary', 'vfcTO', 'vfcV3ab', 'vfcVO']]
            .stack()
        )
        for (cluster, epoch, split), d in dcd_data.groupby(['cluster', 'epoch', 'split']):
            X = pd.pivot_table(data=d.reset_index(), index='subject', columns='latency', values=0)
            table_to_data_source_file(2, 'C', 'ROC_AUC_%s_%s_%s'%(cluster, epoch, split), X)
        
        dcd_data = (dcd
            .test_roc_auc
            .Lateralized
            .query('signal=="MIDC_split"')
            .loc[:, 
                ['JWG_IPS_PCeS', 'JWG_M1', 'JWG_aIPS', 'vfcEarly', 
                 'vfcFEF', 'vfcIPS01', 'vfcIPS23', 'vfcLO', 'vfcPHC', 
                 'vfcPrimary', 'vfcTO', 'vfcV3ab', 'vfcVO']]
            .stack()
        )
        for (cluster, epoch, split), d in dcd_data.groupby(['cluster', 'epoch', 'split']):
            X = pd.pivot_table(data=d.reset_index(), index='subject', columns='latency', values=0)
            table_to_data_source_file(2, 'C-Lateralized', 'ROC_AUC_%s_%s_%s'%(cluster, epoch, split), X)
            


        add_letter(plt.gcf(), gs[2,0], 'C', x=-0.08, y=1.1)

        ax = subplot(gs[3,0])
        ax.set_xlim([0,1])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.text(0.5, 0.5, 'Association\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center', 
            color=(0.75, 0.75, 0.75))
        plt.text(0.075, 0.5, 'Sensory\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center',
            color=(0.75, 0.75, 0.75))
        plt.text(1-0.075, 0.5, 'Motor\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center',
            color=(0.75, 0.75, 0.75))
        plt.arrow( 0.4, 0.5, -0.2, 0, fc=(0.75, 0.75, 0.75), ec=(0.75, 0.75, 0.75),
             head_width=0.5, head_length=0.01 )
        plt.arrow( 0.6, 0.5, 0.2, 0, fc=(0.75, 0.75, 0.75), ec=(0.75, 0.75, 0.75),
             head_width=0.5, head_length=0.01 )
        sns.despine(ax=ax, left=True, bottom=True)
    # plt.tight_layout()
    #savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/2_figure_2.pdf",
    #    dpi=1200
    #)
    return df, stats, dcd


def figureS2(df=None, stats=False, aspect="auto"):  # 0.01883834992799947):
    """
    Plot TFRs underneath each other.
    """
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_stats_20190516.pickle",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190516.hdf"
        )

    #if dcd is None:
    #    dcd = dp.get_decoding_data()
    palette = _stream_palette()

    with mpl.rc_context(rc=rc):
        fig = figure(figsize=(7.5, 10))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(3, 1)

        srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            gs=gs[1, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.67, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-25, 0, 25],
            drawedges=False,
            orientation="horizontal",
            label="% Power change",
        )
        cb.outline.set_visible(False)
        

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["stimulus"],
            flip_cbar=False,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        add_letter(plt.gcf(), gs[1,0], 'B', x=-0.18, y=1.1)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.1)
        """

        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "Reds", "CONF_unsigned": "Greens", "CONF_signed": "Blues"},
            {
                # "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[2, 0],
            title_palette=palette,
        )
        plotter.plot(aspect="auto")

        cax = fig.add_axes([0.81, 0.15, 0.1, 0.0125 / 2])
        cax.plot([-10, -1], [0, 0], "r", label="Choice")
        cax.plot([-10, -1], [0, 0], "b", label="Signed confidence")
        cax.plot([-10, -1], [0, 0], "g", label="Unigned confidence")
        cax.set_xlim([0, 1])
        cax.set_xticks([])
        cax.set_yticks([])
        cax.legend(frameon=False)
        sns.despine(ax=cax, left=True, bottom=True)
        """

    # plt.tight_layout()    
    savefig(
       "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S2_figure_S2.pdf",
    bbox_inches='tight')
    return df, stats


@memory.cache()
def ct(x):
    from mne.stats import permutation_cluster_1samp_test as cluster_test

    return cluster_test(
        x,
        threshold={"start": 0, "step": 0.2},
        connectivity=None,
        tail=0,
        n_permutations=1000,
        n_jobs=4,
    )[2]


def figure_3_test_sig(dcd):
    for cluster in dcd.test_roc_auc.Lateralized.columns:
        if cluster.startswith("NSW"):
            continue
        choice = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="MIDC_split"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        unconf = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="CONF_signed"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        siconf = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="CONF_unsigned"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        u = (choice - unconf).values
        s = (choice - siconf).values
        upchoice = ct(u)
        spchoice = ct(s)
        if (sum(upchoice < 0.05) > 0) or (sum(spchoice < 0.05) > 0):
            print("----->", cluster, upchoice.shape)
            print(
                cluster,
                "Choice vs. unsigned #sig:",
                sum(upchoice < 0.05),
                "#Conf larger:",
                np.sum(u.mean(0)[upchoice < 0.05] < 0),
            )
            print(
                cluster,
                " Choice vs signed #sig:",
                sum(spchoice < 0.05),
                "#Conf larger:",
                np.sum(s.mean(0)[spchoice < 0.05] < 0),
            )


def figureS3(ogldcd=None, pdcd=None):
    if ogldcd is None:
        ogldcd = dp.get_decoding_data(restrict=False, ogl=True)
    if pdcd is None:
        pdcd = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )
    ogldcd = ogldcd.test_roc_auc.Pair
    ogldcd = ogldcd.query('epoch=="stimulus"')
    plt.figure(figsize=(7, 4))
    gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.4)  # ,height_ratios=[3 / 5, 2 / 5])
    with mpl.rc_context(rc=rc):
        # ----> First OGLDD Line plot!
        ogl_t = 1.083
        signal = "MIDC_split"
        low, high = 0.5, 0.8
        # for i, (signal, data) in enumerate(df.groupby(["signal"])):
        sdata = ogldcd.query('signal=="%s"' % signal)
        ax = plt.subplot(gs[0, 0])
        X = pd.pivot_table(
            index="cluster",
            columns="latency",
            values=0,
            data=sdata.stack().reset_index(),
        )
        
        ### Source data files
        for _s, _sd in sdata.stack().reset_index().groupby('cluster'):
            _X = pd.pivot_table(index="subject", columns="latency", data=_sd, values=0)
            table_to_data_source_file('S7', 'A', _s, _X)


        X = X.loc[:, ::2]
            
        txts = []
        idx = np.argsort(X.loc[:, ogl_t])
        sorting = X.index.values[idx]
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap("Reds")
        max_roi = X.loc[:, 1.417].idxmax()

        for j, roi in enumerate(sorting):
            val, color = X.loc[roi, ogl_t], cm(norm(X.loc[roi, ogl_t]))
            if 'V1' == roi:
                print(roi)
                plt.plot(X.columns.values, X.loc[roi, :], color='k', zorder=1000)
            else:
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
            if any([roi == x for x in fig_4_interesting_rois.keys()]) or (
                roi == max_roi
            ):
                try:
                    R = fig_4_interesting_rois[roi]
                except KeyError:
                    R = roi
                if '6d' in roi:
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] + 0.015, R))
                else:
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] - 0.005, R))
        y = np.linspace(0.475, high, 200)
        x = y * 0 - 0.225
        plt.scatter(x, y, c=cm(norm(y)), marker=0)

        plt.title("Spectral power and coarse space\n(hemispheres), whole cortex", fontsize=7)
        plt.xlim([-0.25, 1.4])
        plt.ylim([0.475, high])
        plt.ylabel('AUC', fontsize=7)
        plt.axvline(1.2, color="k", alpha=0.9)
        plt.xlabel("Time", zorder=1, fontsize=7)
        sns.despine(ax=ax)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.35)
        # Now plot brain plots.
        palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, ogl_t]
                for d in X.index.values
            }
        # print(palette)
        palette = pd.DataFrame(palette, index=['average'])
        table_to_data_source_file("S7", 'A', 't1.1_brain', palette)
        img = _get_lbl_annot_img(
            palette,
            low=low,
            high=high,
            views=[["par", "front"], ["med", "lat"]],
            colormap='Reds',
        )

        plt.subplot(gs[1, 0], aspect="equal", zorder=-10)
        plt.imshow(img, zorder=-10)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=plt.gca(), left=True, bottom=True)
        

        # ---> Now PDCD line plot
        pdcd = pdcd.query("target=='response'")
        pdcd_t = 1.1
        low, high = 0.5, 0.8
        ax = plt.subplot(gs[0, 1])
        X = pd.pivot_table(
            index="roi", columns="latency", values="test_roc_auc", data=pdcd
        )

        ### Source data files
        for _s, _sd in pdcd.reset_index().groupby('roi'):
            _X = pd.pivot_table(index="subject", columns="latency", data=_sd, values='test_roc_auc')
            table_to_data_source_file('S7', 'B', _s, _X)
        

        txts = []
        idx = np.argsort(X.loc[:, pdcd_t])
        sorting = X.index.values[idx]
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap('Reds')
        print(sorting)
        for j, roi in enumerate(sorting):
            val, color = X.loc[roi, pdcd_t], cm(norm(X.loc[roi, pdcd_t]))
            if 'V1' in roi:
                print(roi)
                plt.plot(X.columns.values, X.loc[roi, :], color='k')
            else:
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
            if any([roi == x for x in fig_4B_interesting_rois.keys()]):
                R = fig_4B_interesting_rois[roi]
                txts.append(plt.text(1.4, X.loc[roi, 1.4], R))
        y = np.linspace(0.475, high, 200)
        x = y * 0 - 0.225
        plt.scatter(x, y, c=cm(norm(y)), marker=0)
        plt.title('Spectral power and phase, fine space (vertices)\nand coarse space (hemipheres), selected ROIs', fontsize=7)
        plt.xlim([-0.25, 1.4])
        plt.ylim([0.475, high])
        plt.xlabel("Time", fontsize=7)
        plt.axvline(pdcd_t, color="k", alpha=0.9)
        sns.despine(ax=ax)
        add_letter(plt.gcf(), gs[0,1], 'B', x=-0.18, y=1.35)
        # Now plot brain plots.
        
        img = _get_lbl_annot_img(
            palette,
            low=low,
            high=high,
            views=[["par", "front"], ["med", "lat"]],
            colormap='Reds',
        )
        palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, pdcd_t]
                for d in X.index.values
            }
        # print(palette)
        palette = pd.DataFrame(palette, index=['average'])
        table_to_data_source_file("S7", 'B', 't1.1_brain', palette)
        ax = plt.subplot(gs[1, 1], aspect="equal", zorder=-10)
        plt.imshow(img, zorder=-10)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    #savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/S3_figure_S3.pdf",
    #    dpi=1200,
    #    bbox_inches="tight",
    #)
    return ogldcd, pdcd


def _dep_figure4(ogldcd=None, pdcd=None):
    if ogldcd is None:
        ogldcd = dp.get_decoding_data(restrict=False, ogl=True)
    if pdcd is None:
        pdcd = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )
    plt.figure(figsize=(7.5, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1)  # ,height_ratios=[3 / 5, 2 / 5])
    _figure4A(ogldcd, gs=gs[0, 0])
    tight_layout()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_3_figure_3.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.figure(figsize=(7.5, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1)  # , height_ratios=[3 / 5, 2 / 5])
    _figure4B(pdcd, gs=gs[0, 0])
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_4_figure_4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )

    return ogldcd, pdcd


fig_4_interesting_rois = {"4": "M1", "V1": "V1", "2": "IPS/PostCeS", "7PC": "aIPS"}
fig_4B_interesting_rois = {
    "JWG_M1": "M1",
    "vfcPrimary": "V1",
    "JWG_IPS_PCeS": "IPS/PostCeS",
    "JWG_aIPS": "aIPS",
}


def _figure4A(data=None, t=1.083, gs=None):
    import seaborn as sns
    import re
    import matplotlib

    if data is None:
        data = dp.get_decoding_data(restrict=False, ogl=True)
    df = data.test_roc_auc.Pair
    df = df.query('epoch=="stimulus"')

    palettes = {"respones": {}, "unsigned_confidence": {}, "signed_confidence": {}}
    colormaps = {
        "MIDC_split": "Reds",
        "CONF_signed": "Blues",
        "CONF_unsigned": "Greens",
    }
    titles = {
        "MIDC_split": "Choice",
        "CONF_signed": "Signed confidence",
        "CONF_unsigned": "Unsigned confidence",
    }
    v_values = {
        "MIDC_split": (0.5, 0.7),
        "CONF_signed": (0.5, 0.6),
        "CONF_unsigned": (0.5, 0.55),
    }
    high = 0.65
    low = 0.5
    if gs is None:
        plt.figure(figsize=(7.5, 5))
        gs = matplotlib.gridspec.GridSpec(4, 3)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            4, 3, subplot_spec=gs, hspace=0.0, height_ratios=[1, 0.5, 1, 1]
        )

    def corrs(data, t):
        res = {}
        data = data.query("latency==%f" % (t))
        choice = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "MIDC_split").stack().reset_index(),
        )
        uns = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "CONF_unsigned").stack().reset_index(),
        )
        sig = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "CONF_signed").stack().reset_index(),
        )
        res["Ch./Un."] = [
            np.corrcoef(choice.loc[i, :], uns.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        res["Ch./Si."] = [
            np.corrcoef(choice.loc[i, :], sig.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        res["Si./Un."] = [
            np.corrcoef(sig.loc[i, :], uns.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        return pd.DataFrame(res)

    with mpl.rc_context(rc=rc):
        for i, signal in enumerate(colormaps.keys()):
            low, high = v_values[signal]
            # for i, (signal, data) in enumerate(df.groupby(["signal"])):
            sdata = df.query('signal=="%s"' % signal)
            plt.subplot(gs[0, i], zorder=-i)
            X = pd.pivot_table(
                index="cluster",
                columns="latency",
                values=0,
                data=sdata.stack().reset_index(),
            )
            
            print('Subsetting latencies to remove NANs. Check this if decoding is redone')
            X = X.loc[:, ::2]
            
            txts = []
            idx = np.argsort(X.loc[:, t])
            sorting = X.index.values[idx]
            norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
            cm = matplotlib.cm.get_cmap(colormaps[signal])
            max_roi = X.loc[:, 1.417].idxmax()

            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, t], cm(norm(X.loc[roi, t]))
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
                
                if any([roi == x for x in fig_4_interesting_rois.keys()]) or (
                    roi == max_roi
                ):
                    try:
                        R = fig_4_interesting_rois[roi]
                    except KeyError:
                        R = roi
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] - 0.005, R))
            y = np.linspace(0.475, 0.75, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)

            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.75])
            plt.axvline(1.2, color="k", alpha=0.9)
            plt.xlabel("Time", zorder=1)
            if i > 0:
                sns.despine(ax=plt.gca(), left=True)
                plt.yticks([])
            else:
                plt.ylabel("AUC")
                sns.despine(ax=plt.gca())

            palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, t]
                for d in X.index.values
            }
            # print(palette)
            img = _get_lbl_annot_img(
                palette,
                low=low,
                high=high,
                views=[["par", "front"], ["med", "lat"]],
                colormap=colormaps[signal],
            )

            plt.subplot(gs[2, i], aspect="equal", zorder=-10)
            plt.imshow(img, zorder=-10)
            plt.xticks([])
            plt.yticks([])
            sns.despine(ax=plt.gca(), left=True, bottom=True)
        ax = plt.subplot(gs[3, :-1])
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize

        def foo(k):
            norm = Normalize(*v_values[k])
            cmap = get_cmap(colormaps[k])

            def bar(x):
                return cmap(norm(x))

            return bar

        colors = {k: foo(k) for k in colormaps.keys()}
        idx = dp._plot_signal_comp(
            df,
            t,
            None,
            colors,
            "Pair",
            auc_cutoff=0.5,
            ax=ax,
            horizontal=True,
            plot_labels=False,
            color_by_cmap=True,
        )
        signals, subjects, areas, _data = dp.get_signal_comp_data(df, t, "stimulus")
        choice = _data[0, :]
        unsign = _data[1, :]
        signed = _data[2, :]
        _t, _p = ttest_rel(choice.T, unsign.T)
        print("# of Rois where unsigned>choice:", np.sum(_t[_p < 0.05] < 0))
        _t, _p = ttest_rel(choice.T, signed.T)
        print("# of Rois where signed>choice:", np.sum(_t[_p < 0.05] < 0))
        ax.set_xlabel("ROI")
        ax.set_ylabel("AUC")
        ax.set_title("")
        ax.set_ylim([0.49, 0.7])
        y = np.linspace(0.49, 0.7, 250)
        x = y * 0 - 3
        sns.despine(ax=ax, bottom=True)
        o = corrs(df, t=t).stack().reset_index()
        o.columns = ["idx", "Comparison", "Correlation"]
        # o.Comparison.replace({'Ch./Un.':r'\textcolor{red}{Ch./Un.}'}, inplace=True)

        ax = plt.subplot(gs[3, -1])
        sns.stripplot(
            x="Comparison",
            y="Correlation",
            color="k",
            alpha=0.75,
            dodge=True,
            jitter=True,
            ax=ax,
            data=o,
            order=["Ch./Si.", "Ch./Un.", "Si./Un."],
        )
        ax.set_xticklabels(
            ["Choice vs\nSigned", "Choice vs.\nUnsigned", "Signed vs.\nUnsigned"]
        )
        ax.set_xlabel("")
        print(
            "Corr, choice vs signed:",
            o.query('Comparison=="Ch./Si."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Ch./Si."').Correlation, 0),
        )
        print(
            "Corr, choice vs unsigned:",
            o.query('Comparison=="Ch./Un."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Ch./Un."').Correlation, 0),
        )
        print(
            "Corr, signed vs unsigned:",
            o.query('Comparison=="Si./Un."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Si./Un."').Correlation, 0),
        )
        y = o.query('Comparison=="Ch./Si."').Correlation.mean()
        plot([-0.15, 0.15], [y, y], "k")
        y = o.query('Comparison=="Ch./Un."').Correlation.mean()
        plot([1 - 0.15, 1 + 0.15], [y, y], "k")
        y = o.query('Comparison=="Si./Un."').Correlation.mean()
        plot([2 - 0.15, 2 + 0.15], [y, y], "k")
        sns.despine(ax=ax, bottom=True)
    tight_layout()
    return data


def _figure4B(df=None, gs=None, t=1.1):
    import seaborn as sns
    import re
    import matplotlib

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )

    colormaps = {
        "response": "Reds",
        "signed_confidence": "Blues",
        "unsigned_confidence": "Greens",
    }
    titles = {
        "response": "Choice",
        "signed_confidence": "Signed confidence",
        "unsigned_confidence": "Unsigned confidence",
    }
    v_values = {
        "response": (0.5, 0.7),
        "signed_confidence": (0.5, 0.6),
        "unsigned_confidence": (0.5, 0.55),
    }
    if gs is None:
        plt.figure(figsize=(7.5, 5))
        gs = matplotlib.gridspec.GridSpec(2, 3)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs, hspace=0.3
        )
    with mpl.rc_context(rc=rc):
        for i, (signal, data) in enumerate(df.groupby(["target"])):
            low, high = v_values[signal]
            plt.subplot(gs[0, i], zorder=-i)
            X = pd.pivot_table(
                index="roi", columns="latency", values="test_roc_auc", data=data
            )

            txts = []
            idx = np.argsort(X.loc[:, t])
            sorting = X.index.values[idx]
            norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
            cm = matplotlib.cm.get_cmap(colormaps[signal])

            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, t], cm(norm(X.loc[roi, t]))
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
                if any([roi == x for x in fig_4B_interesting_rois.keys()]):
                    R = fig_4B_interesting_rois[roi]
                    txts.append(plt.text(1.4, X.loc[roi, 1.4], R))
            y = np.linspace(0.475, 0.75, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)
            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.75])
            plt.xlabel("Time")
            plt.axvline(t, color="k", alpha=0.9)
            if i > 0:
                sns.despine(ax=plt.gca(), left=True)
                plt.yticks([])
            else:
                plt.ylabel("AUC")
                sns.despine(ax=plt.gca())

            palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, t]
                for d in X.index.values
            }

            img = _get_lbl_annot_img(
                palette,
                low=low,
                high=high,
                views=[["par", "front"], ["med", "lat"]],
                colormap=colormaps[signal],
            )
            plt.subplot(gs[1, i], aspect="equal", zorder=-10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            sns.despine(ax=plt.gca(), left=True, bottom=True)

    tight_layout()
    return df


### This is figure 3 in paper.
def figure5(
    ssd=None, idx=None, oglssd=None, oglidx=None, brain=None, integration_slice=def_ig
):
    measure = 'test_corr'
    if ssd is None:
        ssd = dp.get_ssd_data(restrict=False)
    if idx is None:
        idx = dp.get_ssd_idx(ssd[measure], integration_slice=integration_slice)
    if oglssd is None:
        oglssd = dp.get_ssd_data(restrict=False, ogl=True)
    if oglidx is None:
        oglidx = dp.get_ssd_idx(
            oglssd[measure], integration_slice=integration_slice, pair=True
        )
    if "post_medial_frontal" in ssd[measure].columns:
        del ssd[measure]["post_medial_frontal"]
    if "vent_medial_frontal" in ssd[measure].columns:
        del ssd[measure]["vent_medial_frontal"]
    if "ant_medial_frontal" in ssd[measure].columns:
        del ssd[measure]["ant_medial_frontal"]

    plt.figure(figsize=(8, 11))
    gs = matplotlib.gridspec.GridSpec(
        10, 3, height_ratios=[0.9, 0.9, 0.41, 0.6, 0.25, 0.8, 0.35, 0.5, 0.45, 0.8], hspace=0.1
    )

    with mpl.rc_context(rc=rc):
        # First plot individual lines
        ax = subplot(gs[0, 0])
        plot_per_sample_resp(
            plt.gca(), 
            ssd[measure].Averaged, 
            #ssd[measure].Pair, 
            "vfcPrimary", 
            "V1", 
            integration_slice,
            [-0.05, 0.35]
        )
        legend([], frameon=False)
        xl = plt.xlim()
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        y = np.linspace(0, 0.05, 250)
        for i in np.arange(0, 1, 0.1):            
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
            
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax = subplot(gs[0, 1])
        plot_per_sample_resp(
            plt.gca(),
            #ssd[measure].Lateralized,
            ssd[measure].Pair,
            "JWG_IPS_PCeS",
            "IPS/PostCeS",
            integration_slice,
            [-0.05, 0.35],

        )
        plt.ylabel('')
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')        
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax.set_ylabel("")
        legend([], frameon=False, fontsize=7)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.2)
        ax = subplot(gs[0, 2])
        plot_per_sample_resp(
            plt.gca(),
            #ssd[measure].Lateralized,
            ssd[measure].Pair,
            "JWG_M1",
            "M1-hand",
            integration_slice,
            [-0.05, 0.35],
        )
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        y = np.linspace(0, 0.025, 100)
        norm = matplotlib.colors.Normalize(vmin=-0.025, vmax=0.025)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0, zorder=1)
        #plt.scatter(-0.115+0*y, y, c=cm(norm(-y)), marker=0, zorder=0)
        plt.xlim(xl)
        ax.set_ylabel("")

        legend('', frameon=False)
        sns.despine()


        # Now plot hull curves
        ax = subplot(gs[1, 0])
        plot_per_sample_resp(
            plt.gca(), ssd[measure].Averaged, "vfcPrimary", "", integration_slice,
            [-0.05, 0.35],
            hull=True, acc=True, sig=True
        )
        legend([], frameon=False)
        xl = plt.xlim()
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        y = np.linspace(0, 0.05, 250)
        for i in np.arange(0, 1, 0.1):            
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
            
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax = subplot(gs[1, 1])
        plot_per_sample_resp(
            plt.gca(),
            ssd[measure].Lateralized,
            #ssd[measure].Pair,
            "JWG_IPS_PCeS",
            "",
            integration_slice,
            [-0.05, 0.35],
            hull=True, acc=True, sig=True,
        )

        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')        
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax.set_ylabel("")
        legend([], frameon=False, fontsize=7)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.2)
        ax = subplot(gs[1, 2])
        plot_per_sample_resp(
            plt.gca(),
            ssd[measure].Lateralized,
            #ssd[measure].Pair,
            "JWG_M1",
            "",
            integration_slice,
            [-0.05, 0.35],
            hull=True,
            acc=True, sig=True
        )
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        y = np.linspace(0, 0.025, 100)
        norm = matplotlib.colors.Normalize(vmin=-0.025, vmax=0.025)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0, zorder=1)
        #plt.scatter(-0.115+0*y, y, c=cm(norm(-y)), marker=0, zorder=0)
        plt.xlim(xl)
        ax.set_ylabel("")

        legend(frameon=False)
        sns.despine()


        _figure5A(oglssd, oglidx, gs[3, :])
        add_letter(plt.gcf(), gs[3,:], 'B', y=1.4)
        # _figure5C(ssd.test_slope.Averaged, oglssd.test_slope.Pair, gs=gs[3,:])
        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/3_figure_3_paired.pdf",
        #    dpi=1200,
        #    bbox_inches="tight",
        #)

        
    return ssd, idx, brain


def _figure5A(ssd, idx, gs, integration_slice=def_ig):
    import seaborn as sns
    import matplotlib

    gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs, 
        width_ratios=[0.5, 1, 1, 1],
        height_ratios=[1, 0.075])
    with mpl.rc_context(rc=rc):
        dv = 10

        ax = plt.subplot(gs[0, 1])        
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD for k, vals in m.iterrows()}        
        img = _get_img(palette, low=dv + -0.2, high=dv + 0.2, views=["lat", "med"])
        plt.imshow(img, aspect="equal")        
        plt.xticks([])
        plt.yticks([])
        plt.title("Sample\ncontrast decoding", fontsize=7)
        
        palette = pd.DataFrame(palette, index=['average'])
        table_to_data_source_file(3, 'D', 'sample_contrast_encoding', palette)
        sns.despine(ax=ax, left=True, right=True, bottom=True)

        cax = plt.subplot(gs[1, 1])
        norm = mpl.colors.Normalize(vmin=0,vmax=0.2)
        sm = plt.cm.ScalarMappable(
            cmap=truncate_colormap(plt.get_cmap("RdBu_r"), 0.5, 1),
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[0, 0.1, 0.2], drawedges=False, shrink=0.6)
        cbar.ax.set_xticklabels(
            ['0', '0.1', '0.2'], fontsize=7)


        ax = plt.subplot(gs[0, 2])
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD_acc_contrast for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.2, high=dv + 0.2, views=["lat", "med"])

        palette = pd.DataFrame(palette, index=['average'])
        table_to_data_source_file(3, 'D', 'accumulated_contrast_encoding', palette)
        plt.imshow(img, aspect="equal")

        #plt.colorbar(sm)        
        plt.xticks([])
        plt.yticks([])
        plt.title("Accumulated\ncontrast decoding", fontsize=7)
        sns.despine(ax=ax, left=True, right=True, bottom=True)

        cax = plt.subplot(gs[1, 2])
        norm = mpl.colors.Normalize(vmin=0,vmax=0.2)
        sm = plt.cm.ScalarMappable(
            cmap=truncate_colormap(plt.get_cmap("RdBu_r"), 0.5, 1),
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[0, 0.1, 0.2], drawedges=False, shrink=0.8)
        cbar.ax.set_xticklabels(
            ['0', '0.1', '0.2'], fontsize=7)
        ax = plt.subplot(gs[0, 3])

        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSDvsACC for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.1, high=dv + 0.1, views=["lat", "med"])
        plt.imshow(img, aspect="equal")

        palette = pd.DataFrame(palette, index=['average'])
        table_to_data_source_file(3, 'D', 'encoding_difference', palette)

        norm = mpl.colors.Normalize(vmin=-0.1,vmax=0.1)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        #plt.colorbar(sm, orientation='horizontal')
        plt.xticks([])
        plt.yticks([])
        plt.title("Sample - Accumulated\ncontrast decoding", fontsize=7)
        sns.despine(ax=ax, left=True, right=True, bottom=True)
        cax = plt.subplot(gs[1, 3])
        norm = mpl.colors.Normalize(vmin=-0.1,vmax=0.1)
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[-0.1, 0, 0.1], drawedges=False, shrink=0.8)
        cbar.ax.set_xticklabels(
            ['-0.1\naccumulated\ncontrast enc.', '0', '0.1\nsample\ncontrast enc.'], fontsize=7)
    return ssd, idx


def figureS4(xscores=None, gs=None):
    from scipy.stats import ttest_rel, ttest_1samp
    from mne.stats import fdr_correction

    if xscores is None:
        import pickle

        xscores = (
            pickle.load(
                open(
                    "/Users/nwilming/u/conf_analysis/results/all_areas_gamma_Xarea_stim_latency.pickle",
                    "rb",
                )
            )["scores"]
            .set_index(["latency", "subject", "motor_area"], append=True)
            .stack()
            .reset_index()
        )
        xscores.columns = [
            "del",
            "latency",
            "subject",
            "motor_area",
            "comparison",
            "corr",
        ]
        xscores = xscores.iloc[:, 1:]

    colors = ["windows blue", "amber", "faded green", "dusty purple"]

    if gs is None:
        figure(figsize=(7.5, 1.5))
        gs = matplotlib.gridspec.GridSpec(1, 5, width_ratios=[1, 0.4, 1, 1, 1])
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs, width_ratios=[1, 0.35, 1, 1, 1]
        )
    area_names = ["aIPS", "IPS/PostCeS", "M1-hand"]
    area_colors = _stream_palette()
    for i, signal in enumerate(["JWG_aIPS", "JWG_IPS_PCeS", "JWG_M1"]):
        sd = xscores.query('motor_area=="%s" & (0<=latency<=0.2)' % signal)

        yc = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="corr"'),
        )
        y1smp = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="1smp_corr"'),
        )
        yint = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="integrator_corr"'),
        )
        t, p = ttest_rel(yc, y1smp)

        t, pvsnull = ttest_1samp(yc, 0)
        ax = subplot(gs[0, i + 2])

        ax.plot(
            yc.columns.values,
            (yc.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[0]],
            label="Weighted samples",
        )
        ax.set_ylim([-0.01, 0.05])
        ax.plot(
            y1smp.columns.values,
            (y1smp.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[1]],
            label="Last sample",
        )
        ax.plot(
            yint.columns.values,
            (yint.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[2]],
            label="Integrator",
        )
        id_cor, _ = fdr_correction(p)  # <0.05
        id_unc = pvsnull < 0.05
        draw_sig(ax, np.arctanh(yc - y1smp), p=0.05, fdr=False, lw=2, color=sns.xkcd_rgb[colors[0]])
        #draw_sig(ax, yc, fdr=False, lw=2, color=sns.xkcd_rgb[colors[0]])
        draw_sig(ax, np.arctanh(yint - y1smp), p=0.05, fdr=False, lw=2, y=0.0015, color=sns.xkcd_rgb[colors[2]])
       # draw_sig(ax, yint, fdr=False, lw=2, y=0.0015, color=sns.xkcd_rgb[colors[2]])
        
        title(area_names[i], fontsize=7, color=area_colors[signal])
        if i == 0:
            ax.set_xlabel("Time after sample onset")
            ax.set_ylabel("Correlation")
            sns.despine(ax=ax)
        elif i == 2:            
            ax.legend(frameon=False, loc='center', bbox_to_anchor=(1.7, 0.75))
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
        else:
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
            
    ax = subplot(gs[0, 0])
    # img = _get_lbl_annot_img({'vfcPrimary':0.31, 'JWG_M1':1, 'JWG_aIPS':0.8, 'JWG_IPS_PCeS':0.9},
    #    low=0.1, high=1, views=[["par"]], colormap='viridis')
    img = _get_palette(
        {
            "vfcPrimary": area_colors["vfcPrimary"],
            "JWG_M1": area_colors["JWG_M1"],
            "JWG_aIPS": area_colors["JWG_aIPS"],
            "JWG_IPS_PCeS": area_colors["JWG_IPS_PCeS"],
        },
        views=[["par"]],
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib import patches

    style = "Simple,tail_width=0.5,head_width=4,head_length=8"

    a3 = patches.FancyArrowPatch(
        (501, 559),
        (295, 151),
        connectionstyle="arc3,rad=.25",
        **dict(arrowstyle=style, color=area_colors["JWG_aIPS"])
    )
    a2 = patches.FancyArrowPatch(
        (501, 559),
        (190, 229),
        connectionstyle="arc3,rad=.15",
        **dict(arrowstyle=style, color=area_colors["JWG_IPS_PCeS"])
    )
    a1 = patches.FancyArrowPatch(
        (501, 559),
        (225, 84),
        connectionstyle="arc3,rad=.55",
        **dict(arrowstyle=style, color=area_colors["JWG_M1"])
    )
    ax.add_patch(a3)
    ax.add_patch(a2)
    ax.add_patch(a1)
    sns.despine(ax=ax, left=True, bottom=True)
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S4_figure_S4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )


def _figure5C(ssd, oglssd, gs=None):
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, 4)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs, width_ratios=[1, 1, 0.5, 0.5]
        )
    from conf_analysis.behavior import kernels, empirical
    from scipy.stats import ttest_1samp
    from mne.stats import fdr_correction
    import pickle

    # dz = empirical.get_dz()
    data = empirical.load_data()
    K = data.groupby("snum").apply(kernels.get_decision_kernel)
    # K = dp.extract_kernels(data, contrast_mean=0.5, include_ref=True).T
    C = (
        data.groupby("snum")
        .apply(kernels.get_confidence_kernel)
        .stack()
        .groupby("snum")
        .mean()
    )

    ax = subplot(gs[0, -1])
    ax.plot(K.mean(0), color="k", label="Choice")  # -K.mean(0).mean()
    ax.plot(C.mean(0), color=(0.5, 0.5, 0.5), label="Confidence")  # -C.mean(0).mean()
    ax.legend(frameon=False, bbox_to_anchor=(0.3, 1), loc="upper left")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Behavior")
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_title('behavior')
    # ax.set_xticks(np.arange(10))
    # ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)
    ax = subplot(gs[0, -2])
    ex_kernels = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/example_kernel_vfcPrimary.pickle",
            "rb",
        )
    )
    ex_choice = ex_kernels["choice"]
    ax.plot((ex_choice.mean(0)), color="k", label="Choice")  # -ex_choice.mean()
    ex_conf = ex_kernels["conf"]
    ax.plot(
        (ex_conf.mean(0)), color=(0.5, 0.5, 0.5), label="Confidence"
    )  # ex_conf.mean()
    # ax.legend(frameon=False, bbox_to_anchor= (0.3, 1), loc='upper left')
    # ax.set_title('V1')
    ax.set_xlabel("Sample")
    ax.set_ylabel("V1")
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xticks(np.arange(10))
    # ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)

    cck = pd.read_hdf(
        "/Users/nwilming/u/conf_analysis/results/choice_kernel_correlations.hdf"
    )
    cm = cck.groupby("cluster").mean()
    K_t_palette = dict(cm.choice_corr)
    C_t_palette = dict(cm.conf_corr)
    ax = subplot(gs[0, 0])
    voffset = 1
    low = -0.6
    high = 0.6
    img = _get_lbl_annot_img(
        {k: v + voffset for k, v in K_t_palette.items()},
        views=["lat", "med"],
        low=voffset + low,
        high=voffset + high,
        thresh=None,
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Choice kernel correlation")
    sns.despine(ax=ax, left=True, bottom=True)

    ax = subplot(gs[0, 1])
    img = _get_lbl_annot_img(
        {k: v + voffset for k, v in C_t_palette.items()},
        views=["lat", "med"],
        low=voffset + low,
        high=voffset + high,
        thresh=None,
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Corr. w Conf. kernel")
    sns.despine(ax=ax, left=True, bottom=True)
    return  # K_p_palette, K_t_palette


figure_6colors = sns.color_palette(n_colors=3)
figure_6colors = {
    "AFcorr": figure_6colors[0],
    "DCD": '#db4514', #'#01a56c', #figure_6colors[1],
    "AF-CIFcorr": figure_6colors[2],
    "lAFcorr": figure_6colors[0],
    "lAF-CIFcorr": figure_6colors[2],
    "AccDecode":'#6c01a5'
}


### This figure got killed
def figure6():
    plt.figure(figsize=(8, 4.6))
    gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], hspace=0.25)
    with mpl.rc_context(rc=rc):
        subplot(gs[0,:], zorder=10)
        _figure6_comics(N=10000000)
        add_letter(gcf(), gs[0,:], 'A', y=0.95)
        _figure6A(gs=gs[1, :])
        #_figure6B(gs=gs[2, :])        
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/4_figure_4.pdf",
            dpi=1200,
            bbox_inches="tight",
        )


def _figure6_comics(N=1000):        
    with mpl.rc_context(rc=rc):
        ax = cp_sim(N=N)
        #text(
        #    9.5,
        #    0.77,
        #    "No sensory adaptation",
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        for i in list(range(0, 10)) + list(range(10, 20)) + list(range(20, 30)):
            plot([i, i], [0.5, 0.725], color='gray', alpha=0.5, lw=0.5, zorder=-100)
        text(4.5, 0.75, "(i) Constant sensory gain,\nperfect accumulation", horizontalalignment="center")
        text(
            14.5, 0.75, "(ii) Constant sensory gain,\nbounded accumulation", horizontalalignment="center"
        )
        #plot([4.5, 4.5], [0.75, 0.7725], 'k', lw=1)
        #plot([4.5, 6], [0.7725, 0.7725], 'k', lw=1)
        #plot([13, 14.5], [0.7725, 0.7725], 'k', lw=1)
        #plot([14.5, 14.5], [0.75, 0.7725], 'k', lw=1)
        
        #text(
        #    24.5,
        #    0.77,
        #    "With sensory adaptation",
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        text(24.5, 0.75, "(iii) Sensory adaptation,\nperfect accumulation", horizontalalignment="center")
        #text(
        #    34.5, 0.71, "Bounded accumulation", horizontalalignment="center"
        #)   
        shift = 20
        #plot([4.5+shift, 4.5+shift], [0.75, 0.7725], 'k', lw=1)
        #plot([4.5+shift, 5.7+shift], [0.7725, 0.7725], 'k', lw=1)
        #plot([13.3+shift, 14.5+shift], [0.7725, 0.7725], 'k', lw=1)
        #plot([14.5+shift, 14.5+shift], [0.75, 0.7725], 'k', lw=1)
        
        ax.legend(frameon=False, bbox_to_anchor=(1.07, -.15), loc=0)     
        ax.set_xlim([0, 30])

@memory.cache()
def _get_response(
    fluct, bounded=False, early_noise=0.4, late_noise=0, weights=np.array([1] * 10)
):
    N = fluct.shape[0]
    from conf_analysis.behavior import kernels
    def bounded_acc(fluct, bound=0.95):
        responses = fluct[:, 0] * np.nan
        cumsums = fluct.cumsum(1)
        for col in np.arange(fluct.shape[1]):
            id_pos = (cumsums[:, col] > bound) & np.isnan(responses)
            id_neg = (cumsums[:, col] < -bound) & np.isnan(responses)
            responses[id_pos] = 1
            responses[id_neg] = -1
        id_pos = (cumsums[:, col] > 0) & np.isnan(responses)
        id_neg = (cumsums[:, col] < 0) & np.isnan(responses)
        responses[id_pos] = 1
        responses[id_neg] = -1
        return responses

    early_noise = early_noise * (np.random.normal(size=(N, 10)))
    late_noise = early_noise * (np.random.normal(size=(N, 10)))
    correct = fluct.mean(1) > 0

    internal_flucts = (fluct + early_noise) * weights
    if bounded:
        resp = bounded_acc(internal_flucts + late_noise, bound=bounded)
    else:
        resp = ((internal_flucts + late_noise).mean(1) > 0).astype(int)
        resp[resp == 0] = -1
    accuracy = np.mean((resp == 1) == correct)
    behavioral_kernel = kernels.kernel(fluct, resp.astype(int))[0]
    cp = kernels.kernel(internal_flucts, resp.astype(int))[0]
    contrast = weights-weights.mean()
    return accuracy, behavioral_kernel, cp, contrast.ravel()


@memory.cache()
def _get_fluct(t, N):
    return 0.1 * (np.random.normal(size=(N, 10))) + t

def cp_sim(t=0.1, noise_mag=0.43, N=15000):
    fluct = _get_fluct(t, N)
    yl = [0.499, 0.8]
    
    const = np.linspace(1, 0, 10)[np.newaxis, :] * 0 + 1
    weights = np.linspace(1, 0.1, 10)[np.newaxis, :]
    ax = gca()
    for i, (bound, weights, enfudge) in enumerate(
        zip([False, True, False,],#True], 
            [const, const, weights,],# weights], 
            [0.43, 0.385, 0.286,])# 0.295])
    ):
        x = np.arange(10)+i*10
        
        accuracies = []
        kernels = []
        contrasts = []
        if bound:
            for b in [0.95-0.5, 0.95, 0.95+0.5]:
                accuracy, behavioral_kernel, cp, contrast = _get_response(
                    fluct, bounded=b, weights=weights, early_noise=enfudge                
                )
                accuracies.append(accuracy)
                kernels.append(behavioral_kernel)
                contrasts.append(contrast)
        elif weights is const:

            accuracy, behavioral_kernel, cp, contrast = _get_response(
                fluct, bounded=bound, weights=weights, early_noise=enfudge
            )
            accuracies.append(accuracy)
            kernels.append(behavioral_kernel)
            contrasts.append(contrast)
        else:
            for w in [0.75, 1, 1.25]:
                ws = np.linspace(w, 0.1, 10)[np.newaxis, :]
                accuracy, behavioral_kernel, cp, contrast = _get_response(
                    fluct, bounded=bound, weights=ws, early_noise=enfudge
                )
                accuracies.append(accuracy)
                kernels.append(behavioral_kernel)
                contrasts.append(contrast)
        
        for j,(behavioral_kernel, accuracy, contrast) in enumerate(zip(kernels, accuracies, contrasts)):
            lw = 1
            if (len(kernels) == 3) and not (j==1):
                lw = 0.5
            if i==0:
                plot(x, behavioral_kernel+0.05, lw=lw, color="k", label="Behavioral kernel")
                #plot(x, cp, color=figure_6colors["AFcorr"], label="V1 (gamma band) kernel")
                if contrast[-1] < 0:
                    plot(x, 0.1*contrast+0.6, lw=lw, color=figure_6colors["DCD"], label="V1 contrast")
                else:
                    plot(x, contrast+0.6, lw=lw, color=figure_6colors["DCD"], label="V1 contrast")
            else:
                plot(x, behavioral_kernel+0.05, lw=lw, color="k")
                #plot(x, cp, color=figure_6colors["AFcorr"])
                if contrast[-1] < 0:
                    plot(x, 0.1*contrast+0.6, lw=lw, color=figure_6colors["DCD"])
                else:
                    plot(x, contrast+0.6, lw=lw, color=figure_6colors["DCD"])
        ylim(yl)
        #text(0+i*10+0.5, 0.509, 'Accuracy=%0.3f'%np.around(accuracy, 3))
        text(0+i*10+4.5, 0.46, 'Time (s)', horizontalalignment='center')
        yticks([])
    #xticks([0, 9, 10, 19, 20, 29, 30, 39], [0, 1, 0, 1, 0, 1, 0, 1])
    xticks([0, 9, 10, 19, 20, 29,], [0, 1, 0, 1, 0, 1,])
    sns.despine(ax=ax, left=True, bottom=True)
    ylabel('AUC /      \nDecoding precision (a.u)')
    plot([0, 0], [0.5, 0.725], 'k')
    plot([0, 9], [0.5, 0.5], 'k')
    plot([10, 19], [0.5, 0.5], 'k')
    plot([20, 29], [0.5, 0.5], 'k')
    #plot([30, 39], [0.5, 0.5], 'k')
    return ax


def _figure6A(cluster="vfcPrimary", freq="gamma", lowfreq=10, gs=None, label=True):
    import matplotlib
    import seaborn as sns
    import pickle

    fs = {
        "gamma": "/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle",
        "alpha": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle",
        "beta": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle",
    }
    try:
        a = pickle.load(open(fs[freq], "rb"))
    except KeyError:
        fname = (
            "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
    clpeaks = get_cl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    M1decoding = clpeaks["vfcPrimary"]

    v1dcdslopes = v1decoding.apply(lambda x: linregress(np.arange(10), x)[0])
    print(
        "V1 decoding slopes (mean, p, t):",
        np.around(np.mean(v1dcdslopes), 3),
        ttest_1samp(v1dcdslopes, 0),
    )

    fname = (
        "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
        % lowfreq
    )
    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]

    colors = figure_6colors

    # plt.figure(figsize=(12.5, 5.5))
    if gs is None:
        figure(figsize=(10.5, 3))
        gs = matplotlib.gridspec.GridSpec(
            1, 3, width_ratios=[1, 1, 0.1], wspace=0.35
        )
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs, width_ratios=[1, 1, 0.1], wspace=0.35
        )
    # gs = matplotlib.gridspec.GridSpec(1, 4)
    ax = plt.subplot(gs[0, 0])
    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
    alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")

    low_kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query(
        'cluster=="%s"' % cluster
    )
    low_rems = pd.pivot_table(data=low_kernels.query("rmcif==True"), index="subject")
    low_alls = pd.pivot_table(data=low_kernels.query("rmcif==False"), index="subject")

    plotwerr(K, color="k", label="Behavioral kernel", lw=1)
    draw_sig(ax, K, y=-0.001, fdr=True, color="k")
    sns.despine(ax=ax, bottom=False, right=True)
    plt.ylabel("AUC-0.5")
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time (s)")
    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.ylim([-0.003, 0.13])
    for i in range(10):
        plt.axvline(i, color='gray', zorder=-100000, alpha=0.5)
    ax = plt.subplot(gs[0, 1])
    plotwerr(v1decoding.T, label="V1 contrast", color=colors["DCD"], lw=1)
    draw_sig(ax, K, y=-0.001, fdr=True, color=colors["DCD"])
    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding precision (a.u.)")
    plt.ylim([-0.003, 0.13])
    # plt.axhline(0, color='k', lw=1)
    #plt.legend(
    #    frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.2, 1], fontsize=8
    #)
    for i in range(10):
        plt.axvline(i, color='gray', zorder=-100000, alpha=0.5)
    sns.despine(ax=ax, bottom=False, right=True)
    add_letter(plt.gcf(), gs[:,:2], 'B')

    ax = plt.subplot(gs[0, 2])
    nine = np.array([np.corrcoef(x[:9], y[:9])[0,1] for x,y in zip(K.values, clpeaks['vfcPrimary'].values.T)])
    plot(nine*0 + np.random.randn(15)/5, nine, 'ok', alpha=0.5)
    plot([-0.5, 0.5], [np.mean(nine), np.mean(nine)], 'k', lw=2)
    t,p = ttest_1samp(np.arctanh(nine), 0)
    print('Correlation kernel, decoding (m, t, p):', np.mean(nine), t,p)
    ylim([-1, 1])
    xlim([-0.75, 0.75])
    xticks([])
    yticks([-1, -0.5, 0, 0.5, 1])
    ylabel('Correlation behavioral kernel\nand V1 Contrast kernel')
    sns.despine(ax=ax, bottom=True, right=True)
    add_letter(plt.gcf(), gs[:,2], 'C', x=-2.2)

    """
    ax = plt.subplot(gs[0, 2], zorder=1)
    plotwerr(alls, label="V1 kernel (gamma band)", color=colors["AFcorr"])
    #plotwerr(rems, label="Contrast fluctu-\nations removed", color=colors["AF-CIFcorr"])    
    draw_sig(ax, alls, fdr=False, y=-0.0025, color=colors["AFcorr"])
    #draw_sig(ax, rems, y=-0.01125, color=colors["AF-CIFcorr"])
    #draw_sig(ax, alls - rems, y=-0.0125, color="k")

    # plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time")
    plt.ylabel("AUC")
    plt.yticks([0, 0.01, 0.02, 0.03])
    plt.ylim(-0.003, 0.03)
    #yl = plt.ylim()
    # plt.axhline(0, color='k', lw=1)
    #plt.legend(
    #    frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.15, 0.8], fontsize=8
    #)
    sns.despine(ax=ax, bottom=False, right=True)
    #plt.title("Gamma", fontsize=8)
    """


### This is figure 4 in paper.
def figure7():
    plt.figure(figsize=(8, 4.5))
    gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], hspace=0.4)
    with mpl.rc_context(rc=rc):                
        _figure7A(gs=gs[0, :])
        _figure7B(gs=gs[1, :])
        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/4_figure_4.pdf",
        #    dpi=1200,
        #    bbox_inches="tight",
        #)


def _figure7B(cluster="vfcPrimary", gs=None, ct=False, fdr=False):
    import pickle
    from scipy.stats import linregress, ttest_1samp

    res = []
    clpeaks = get_cl_decoding_peaks()
    oglpeaks = get_ogl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    intdecoding = oglpeaks["3b"]
    m1decoding = clpeaks["JWG_M1"]
    corrs = []
    rescorr = []
    for freq in list(range(1, 10)) + list(range(10, 115, 5)):
        fname = (
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
        ccs, K, kernels, _ = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
        kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
        KK = np.stack(kernels.kernel)
        kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
        rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
        alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
        allsV1 = alls.mean(1).to_frame()
        allsV1.loc[:, "comparison"] = "V1kernel"
        allsV1.loc[:, "freq"] = freq
        allsV1.columns = ["sum", "comparison", "freq"]
        res.append(allsV1)
        remsV1 = rems.mean(1).to_frame()
        remsV1.loc[:, "comparison"] = "V1kernelCIF"
        remsV1.loc[:, "freq"] = freq
        remsV1.columns = ["sum", "comparison", "freq"]
        res.append(remsV1)
        #allsslopes = alls.apply(
        #    lambda x: linregress(np.arange(10), x)[0], axis=1
        #).to_frame()
        allsslopes = alls.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        # alls.mean(1).to_frame()
        allsslopes.loc[:, "comparison"] = "slope"
        allsslopes.loc[:, "freq"] = freq
        allsslopes.columns = ["sum", "comparison", "freq"]
        res.append(allsslopes)

        remsslopes = rems.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        # alls.mean(1).to_frame()
        remsslopes.loc[:, "comparison"] = "rems_slope"
        remsslopes.loc[:, "freq"] = freq
        remsslopes.columns = ["sum", "comparison", "freq"]
        res.append(remsslopes)

        # Compute correlation between DCD kernel and ACC kernel
        freqs = v1decoding.columns.values
        for lag in range(-2, 3):
            # Example:
            # high level (motor): [N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = 1
            highlevel_slice = slice(np.abs(lag), 10)
            lowlevel_slice = slice(0, (10 - np.abs(lag)))
            if lag < 0:
                highlevel_slice, lowlevel_slice = lowlevel_slice, highlevel_slice
            x = np.arange(10)
            # Example
            # high level (motor): [ N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = -1
            for i in range(15):
                rescorr.append(
                    {
                        "freq": freq,
                        "V1DCD-AFcorr": np.corrcoef(
                            v1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "M1DCD-AFcorr": np.corrcoef(
                            m1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "VODCD-AFcorr": np.corrcoef(
                            intdecoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "subject": i + 1,
                        "lag": lag,
                    }
                )
    rescorr = pd.DataFrame(rescorr)
    res = pd.concat(res)
    n = 5
    width_ratios = [1, 0.25, 1, 0.25, 0.5]
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, n, width_ratios=width_ratios)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, n, subplot_spec=gs, width_ratios=width_ratios
        )
    ax = plt.subplot(gs[0, 0])
    allv1 = pd.pivot_table(
        data=res.query('comparison=="V1kernel"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    plotwerr(allv1.loc[:, 4:], color=figure_6colors["AFcorr"])
    table_to_data_source_file(4, 'D', 'Kernel_mean_spectrum', allv1)
    cifrem = pd.pivot_table(
        data=res.query('comparison=="V1kernelCIF"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    #plotwerr(cifrem, color=figure_6colors["AF-CIFcorr"])
    # draw_sig(ax,diff, fdr=False, color='k')
    #draw_sig(ax, allv1 - cifrem, conjunction=allv1, y=-0.0012, fdr=fdr, cluster_test=ct, color="k") 
    draw_sig(ax, allv1.loc[:, 4:], fdr=fdr, cluster_test=ct, color=figure_6colors["AFcorr"]) #conjunction=allv1,
    #draw_sig(ax, cifrem, fdr=fdr, cluster_test=ct, y=-0.00065, color=figure_6colors["AF-CIFcorr"])
    #draw_sig(ax, allv1 - cifrem, fdr=False, cluster_test=True, color="k")
    p = ttest_1samp((allv1 - cifrem).values, 0)[1]
    p2 = ttest_1samp(allv1.values, 0)[1]
    print("Sum sig. frequencies:", allv1.columns.values[(p < 0.05) & (p2 < 0.05)])
    xlabel("Frequency (Hz)", fontsize=7)
    ylabel("Kernel sum", fontsize=7)
    yticks([-0.005, 0, 0.005, 0.01])
    #for i in range(10):
    #    ax.axvline(i, color='gray', alpha=0.5, lw=0.5)
    ax.axhline(0, color='gray', zorder=-10)
    xlim([2, xlim()[1]])
    xticks([10, 50, 100])
    sns.despine(ax=ax)

    add_letter(plt.gcf(), gs[0,0], 'D', x=-0.35)

    ax = plt.subplot(gs[0, 2])
    diff = pd.pivot_table(
        data=res.query('comparison=="slope"'),
        index="subject",
        columns="freq",
        values="sum",
    )

    plotwerr(diff.loc[:, 4:])
    table_to_data_source_file(4, 'D', 'Kernel_slope_spectrum', diff)
    p = ttest_1samp(diff.values, 0)[1]
    # print(diff.values.mean(0), p)
    print("Slope sig. frequencies:", diff.columns.values[p < 0.05])
    draw_sig(ax, diff, fdr=fdr, cluster_test=ct, color=figure_6colors["AFcorr"])
    #draw_sig(ax, diff, cluster_test=True, fdr=False, color=figure_6colors["AFcorr"])

    #--> REM SLOPES
    diffr = pd.pivot_table(
        data=res.query('comparison=="rems_slope"'),
        index="subject",
        columns="freq",
        values="sum",
    )

    #plotwerr(diffr, color=figure_6colors["AF-CIFcorr"])    
    #draw_sig(ax, diffr, fdr=fdr, cluster_test=ct, color=figure_6colors["AF-CIFcorr"], y=-0.0002)
    

    #<-- REM SLOPES
    # draw_sig(ax,diff, fdr=True, color='g')
    xlabel("Frequency (Hz)", fontsize=7)
    ylabel("Slope of\nV1 kernel", fontsize=7)
    yticks([-0.002, 0, 0.002])
    xlim([2, xlim()[1]])
    xticks([10, 50, 100])
    sns.despine(ax=ax)
    add_letter(plt.gcf(), gs[0,2], 'E', x=-0.4)
    #for i in range(10):
    #    ax.axvline(i, color='gray', alpha=0.5, lw=0.5)
    ax.axhline(0, color='gray', zorder=-10)

    ### Time lag plot follows
    """
    ax = plt.subplot(gs[0, -1])
    alpha_M1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="M1DCD-AFcorr",
    )
    print(rescorr.query("10<=freq<=16").freq.unique())
    alpha_int1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="VODCD-AFcorr",
    )
    alpha_V1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="V1DCD-AFcorr",
    )
    plotwerr(alpha_M1)
    print("P-values Feedback M1->V1:", ttest_1samp(np.arctanh(alpha_M1), 0))
    draw_sig(ax, np.arctanh(alpha_M1), color=figure_6colors["AFcorr"])
    ylabel("Correlation\nV1 Alpha /  M1 decoding", fontsize=7)
    yticks([0, 0.15, 0.3], fontsize=7)
    xlabel("Lag (number of\nstimulus samples)", fontsize=7)
    xticks([-2, 0, 2], ["-2\nV1 leads", 0, "2\nM1 leads"], fontsize=7)
    """
    ### Replace with correlation between kernels
    #ax = plt.subplot(gs[0, -1])
    #alpha_kernel, gamma_kernel = _get_alpha_gamma_kernel()
    #CCs = np.array([np.corrcoef(alpha_kernel.loc[s,:], gamma_kernel.loc[s, :])[0,1] for s in range(1, 16)])
    #plot(np.random.randn(15)*0.1, CCs, 'ko', alpha=0.5)
    #plot([-0.5, 0.5], [f


def _figure7B_panelD(cluster="vfcPrimary", gs=None, ct=False, fdr=False):
    import pickle
    from scipy.stats import linregress, ttest_1samp
    figure(figsize=(3, 3))
    res = []
    clpeaks = get_cl_decoding_peaks()
    oglpeaks = get_ogl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    intdecoding = oglpeaks["3b"]
    m1decoding = clpeaks["JWG_M1"]
    corrs = []
    rescorr = []
    for freq in list(range(1, 10)) + list(range(10, 115, 5)):
        fname = (
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
        ccs, K, kernels, _ = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
        kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
        KK = np.stack(kernels.kernel)
        kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
        rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
        alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
        allsV1 = alls.mean(1).to_frame()
        allsV1.loc[:, "comparison"] = "V1kernel"
        allsV1.loc[:, "freq"] = freq
        allsV1.columns = ["sum", "comparison", "freq"]
        res.append(allsV1)
        remsV1 = rems.mean(1).to_frame()
        remsV1.loc[:, "comparison"] = "V1kernelCIF"
        remsV1.loc[:, "freq"] = freq
        remsV1.columns = ["sum", "comparison", "freq"]
        res.append(remsV1)
        
        allsslopes = alls.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        
        allsslopes.loc[:, "comparison"] = "slope"
        allsslopes.loc[:, "freq"] = freq
        allsslopes.columns = ["sum", "comparison", "freq"]
        res.append(allsslopes)

        remsslopes = rems.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        
        # alls.mean(1).to_frame()
        remsslopes.loc[:, "comparison"] = "rems_slope"
        remsslopes.loc[:, "freq"] = freq
        remsslopes.columns = ["sum", "comparison", "freq"]
        res.append(remsslopes)

        # Compute correlation between DCD kernel and ACC kernel
        freqs = v1decoding.columns.values
        for lag in range(-2, 3):
            # Example:
            # high level (motor): [N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = 1
            highlevel_slice = slice(np.abs(lag), 10)
            lowlevel_slice = slice(0, (10 - np.abs(lag)))
            if lag < 0:
                highlevel_slice, lowlevel_slice = lowlevel_slice, highlevel_slice
            x = np.arange(10)
            # Example
            # high level (motor): [ N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = -1
            for i in range(15):
                rescorr.append(
                    {
                        "freq": freq,
                        "V1DCD-AFcorr": np.corrcoef(
                            v1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "M1DCD-AFcorr": np.corrcoef(
                            m1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "VODCD-AFcorr": np.corrcoef(
                            intdecoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "subject": i + 1,
                        "lag": lag,
                    }
                )
    rescorr = pd.DataFrame(rescorr)
    res = pd.concat(res)
    n = 5
    
    ax = plt.subplot(111)
    allv1 = pd.pivot_table(
        data=res.query('comparison=="V1kernel"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    plotwerr(allv1/10, color=figure_6colors["AFcorr"])

    cifrem = pd.pivot_table(
        data=res.query('comparison=="V1kernelCIF"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    #plotwerr(cifrem, color=figure_6colors["AF-CIFcorr"])
    # draw_sig(ax,diff, fdr=False, color='k')
    #draw_sig(ax, allv1 - cifrem, conjunction=allv1, y=-0.0012, fdr=fdr, cluster_test=ct, color="k") 
    draw_sig(ax, allv1, fdr=fdr, cluster_test=ct, color=figure_6colors["AFcorr"]) #conjunction=allv1,
    #draw_sig(ax, cifrem, fdr=fdr, cluster_test=ct, y=-0.00065, color=figure_6colors["AF-CIFcorr"])
    #draw_sig(ax, allv1 - cifrem, fdr=False, cluster_test=True, color="k")
    #p = ttest_1samp((allv1 - cifrem).values, 0)[1]
    p2 = ttest_1samp(allv1.values, 0)[1]
    print("Sum sig. frequencies:", allv1.columns.values[(p2 < 0.05)])
    xlabel("Frequency (Hz)", fontsize=7)
    ylabel("Kernel mean", fontsize=7)

    yticks([-0.005/10, 0, 0.005/10, 0.01/10])

    xticks([1, 25, 50, 75, 100], fontsize=6)
    yticks(fontsize=6)
    ax.axhline(0, color='gray', zorder=-10)    
    sns.despine(ax=ax)
    savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/Sx_panelD_from_figure_4.pdf",
            dpi=1200,
            bbox_inches="tight",
        )


def get_kernels_from_file(fname, cluster='vfcPrimary'):
    import pickle
    a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
    alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
    return rems, alls


def _get_alpha_gamma_kernel():
    import pickle
    lowfreq=10
    cluster='vfcPrimary'
    freq='gamma'
    fs = {
        "gamma": "/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle",
        "alpha": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle",
        "beta": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle",
    }
    try:
        a = pickle.load(open(fs[freq], "rb"))
    except KeyError:
        fname = (
            "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]

    fname = (
        "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
        % lowfreq
    )

    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]


    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    low_kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query(
        'cluster=="%s"' % cluster
    )
    return low_kernels.query('rmcif==False').groupby('subject').mean(), kernels.query('rmcif==False').groupby('subject').mean()


def _get_alpha_gamma_kernel_rmcif(cluster='vfcPrimary'):
    import pickle
    lowfreq=10
    
    freq='gamma'
    fs = {
        "gamma": "/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle",
        "alpha": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle",
        "beta": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle",
    }
    try:
        a = pickle.load(open(fs[freq], "rb"))
    except KeyError:
        fname = (
            "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]


    fname = (
        "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
        % lowfreq
    )

    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]


    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    low_kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query(
        'cluster=="%s"' % cluster
    )
    return low_kernels.query('rmcif==True').groupby('subject').mean(), kernels.query('rmcif==True').groupby('subject').mean()


def _figure7A(cluster="vfcPrimary", freq="gamma", lowfreq=10, gs=None, label=True, ct=False, fdr=False):
    import matplotlib
    import seaborn as sns
    import pickle

    fs = {
        "gamma": "/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle",
        "alpha": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle",
        "beta": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle",
    }
    try:
        a = pickle.load(open(fs[freq], "rb"))
    except KeyError:
        fname = (
            "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
    

    clpeaks = get_cl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    M1decoding = clpeaks["vfcPrimary"]

    v1dcdslopes = v1decoding.apply(lambda x: linregress(np.arange(10), x)[0])
    print(
        "V1 decoding slopes (mean, p, t):",
        np.around(np.mean(v1dcdslopes), 3),
        ttest_1samp(v1dcdslopes, 0),
    )

    fname = (
        "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
        % lowfreq
    )
    print('Lowfreq filename:', fname)
    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]

    colors = figure_6colors

    
    if gs is None:
        figure(figsize=(10.5, 3))
        gs = matplotlib.gridspec.GridSpec(
            1, 9, width_ratios=[1, 0.75, 0.25, 1, 1, 0.55, 1, 0.35, 0.35]
        )
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 9, subplot_spec=gs, width_ratios=[1, 0.75, 0.25, 1, 1, 0.55, 1, 0.35, 0.35]
        )
    

    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
    alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")

    low_kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query(
        'cluster=="%s"' % cluster
    )
    low_rems = pd.pivot_table(data=low_kernels.query("rmcif==True"), index="subject")
    low_alls = pd.pivot_table(data=low_kernels.query("rmcif==False"), index="subject")

    ax = plt.subplot(gs[0, :3], zorder=2)
        
    plotwerr(alls, label="Overall kernel", color=colors["AFcorr"])
    table_to_data_source_file(4, 'A', 'Overall_gamma_band', alls)
    plotwerr(rems, label="Residual kernel (contrast\nfluctuations removed)", color=colors["AF-CIFcorr"])#, fontsize=7)
    table_to_data_source_file(4, 'A', 'Residual_gamma_band', alls)

    gamma_k_slopes = alls.T.apply(lambda x: linregress(np.arange(0, 1, 0.1), x)[0])
    print('Mean Gamma kernel slope:', gamma_k_slopes.mean(), ttest_1samp(gamma_k_slopes, 0))
    draw_sig(ax, alls, y=-0.01, color=colors["AFcorr"], cluster_test=ct, fdr=fdr, debug='GAMMA-kernel')
    draw_sig(ax, rems, y=-0.01125, color=colors["AF-CIFcorr"], cluster_test=ct, fdr=fdr)
    draw_sig(ax, alls - rems, y=-0.0125, color="k", cluster_test=ct, fdr=fdr)
    
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"], fontsize=7)
    plt.xlabel("Contrast sample number", fontsize=7)
    plt.ylabel("AUC-0.5", fontsize=7)
    plt.yticks([0, 0.02, 0.05], fontsize=7)
    yl = plt.ylim([-0.035, 0.04])
    
    plt.legend(
        frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.15, 0.8], fontsize=7
    )
    for i in range(10):
        ax.axvline(i, color='gray', alpha=0.5, lw=0.5, zorder=-100)
    ax.axhline(0, color='gray', zorder=-100)
    sns.despine(ax=ax, bottom=False, right=True)
    plt.title("Gamma (45-60Hz)", fontsize=7)

    add_letter(plt.gcf(), gs[0,:3], 'A', y=1.15, x=-0.3)

    ax = plt.subplot(gs[0, 3:5], zorder=1)

    plotwerr(low_alls, ls=":", color=colors["AFcorr"])  # label='V1 kernel',
    table_to_data_source_file(4, 'A', 'Overall_low_frequency_band', low_alls)
    plotwerr(low_rems, ls=":", color=colors["AF-CIFcorr"])       
    table_to_data_source_file(4, 'A', 'Residual_low_frequency_band', low_rems) 

    draw_sig(ax, low_alls, y=-0.01, color=colors["AFcorr"], cluster_test=ct, fdr=fdr, debug='ALPHA-kernel')
    draw_sig(ax, low_rems, y=-0.01125, color=colors["AF-CIFcorr"], cluster_test=ct, fdr=fdr)
    draw_sig(ax, low_alls - low_rems, y=-0.0125, color="k", cluster_test=ct, fdr=fdr)

    plt.legend(
        frameon=False, ncol=1, fontsize=7
    )

    for i in range(10):
        ax.axvline(i, color='gray', alpha=0.5, lw=0.5, zorder=-100)
    ax.axhline(0, color='gray', zorder=10)
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"], fontsize=7)
    plt.xlabel("Contrast sample number", fontsize=7)
    plt.ylim(yl)
    plt.ylabel(None)
    plt.yticks([])
    plt.title("Alpha (10Hz)", fontsize=7)
    

    sns.despine(ax=ax, bottom=False, right=True, left=True)

    rescorr = []
    for i in range(15):
        rescorr.append(
            {
                "AFcorr": np.corrcoef(alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "AFcorr_nine_samples": np.corrcoef(alls.loc[i + 1, :8], K.loc[i + 1, :8])[0, 1],
                "AF-CIFcorr": np.corrcoef(rems.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "V1DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[
                    0, 1
                ],
                "M1DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[
                    0, 1
                ],
                "DCD": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], alls.loc[i + 1, :])[
                    0, 1
                ],
                "lAFcorr": np.corrcoef(low_alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "lAF-CIFcorr": np.corrcoef(low_rems.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "subject": i + 1,
            }
        )        
    rescorr = pd.DataFrame(rescorr)
    rs = rescorr.set_index("subject").stack().reset_index()
    rs.columns = ["subject", "comparison", "correlation"]
    dcdcorr = rs.query('comparison=="DCD"').correlation.values
    print(
        "DCD correlation with behavior K (mean, p, t):",
        np.around(np.mean(dcdcorr), 2),
        ttest_1samp(dcdcorr, 0),
    )
    #yl = plt.ylim()

    #### Baseline AUC Kernel plot follows
    ax = plt.subplot(gs[0, -4])
    kernel = pd.read_hdf("/Users/nwilming/u/conf_analysis/results/nr_high_res_kernels.h5")
    idkk = kernel.columns.values
    idkk = idkk[np.argmin(np.abs(idkk-(-0.1)))]
    print('####', idkk)
    #
    #kd_ = kernel.query('cluster=="vfcPrimary" & hemi=="Averaged"').loc[:, idkk].groupby(['subject', 'freqband']).mean().unstack()
    #kd_ = kernel.query('cluster=="vfcPrimary" & hemi=="Averaged"').loc[:, -0.25:0].groupby(['subject', 'freqband']).mean().unstack()
    kd_ = kernel.query('cluster=="vfcPrimary" & hemi=="Averaged"').loc[:, -0.25:0].groupby(['freqband', 'subject']).mean().mean(1).unstack('freqband')
    kd_mean_ = kd_.mean(0).values
    kd_std_ = kd_.std(0).values
    kd_sem_ = kd_std_ / (15 ** 0.5)
    
    print('$$$$$ Baseline test', ttest_1samp(kd_.values[:, 0], 0))
    print('$$$$$ Baseline test', ttest_1samp(kd_.values[:, 1], 0))
        
    
    plot([-0.2, 0.2], [kd_mean_[0], kd_mean_[0]], 'k-', zorder=100, lw=2)
    plot([1-.2, 1.2], [kd_mean_[1], kd_mean_[1]], 'k-', zorder=100, lw=2)
    sns.stripplot(
        data=kd_.stack().reset_index(),
        x="freqband",
        y=0,
        #order=["AFcorr"],# "AF-CIFcorr"],
        color='gray',
    )
    table_to_data_source_file(4, 'B', 'Baseline_auc', kd_)
    
    print('########:', [kd_mean_[0] - kd_sem_[0], kd_mean_[0] + kd_sem_[0]])
    print(kd_mean_)
    """
    plot([0, 1], kd_mean_, 'ko')
    plot([0, 0], [kd_mean_[0] - kd_sem_[0], kd_mean_[0] + kd_sem_[0]], 'k')
    plot([1, 1], [kd_mean_[1] - kd_sem_[1], kd_mean_[1] + kd_sem_[1]], 'k')
    """
    sns.despine(ax=ax, left=True)
    plt.yticks([])
    plt.xticks([0, 1], ['Gamma', 'Alpha'], rotation=45)
    plt.xlim([-0.25, 1.25])
    plt.ylim(yl)
    plt.axhline(0, color='k', alpha=0.45)
    plt.title("Baseline\nperiod (-250-0ms)", fontsize=7, verticalalignment='center')
    add_letter(plt.gcf(), gs[0,-4], 'B', y=1.15, x=-0.35)

    ##### Panel B Follows.
    ax = plt.subplot(gs[0, -2])
    sns.stripplot(
        data=rs,
        x="comparison",
        y="correlation",
        order=["AFcorr"],# "AF-CIFcorr"],
        palette='gray',
    )
    table_to_data_source_file(4, 'C', 'gamma_correlation_V1_kernel', rs.query('comparison=="AFcorr"').loc[:, ['subject', 'correlation']])

    plt.ylabel("Correlation with\nbehavioral kernel", fontsize=7)
    plt.plot(
        [-0.2, +0.2],
        [rescorr.loc[:, "AFcorr"].mean(), rescorr.loc[:, "AFcorr"].mean()],
        "k", lw=2, 
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    p9 = ttest_1samp(np.arctanh(rescorr.loc[:, "AFcorr_nine_samples"]), 0)
    print(
        "M/T/P corr gamma kernel w/ behavior kernel, first nine samples only:",
        np.around(np.mean(rescorr.loc[:, "AFcorr_nine_samples"]), 2),
        p9,
    )
    if p[1] < 0.001:
        plt.text(0, 0.9, '***', fontsize=12, verticalalignment='center', horizontalalignment='center')

    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(1, 0.95, '*', fontsize=10, verticalalignment='center')
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), np.arctanh(rescorr.loc[:, "AFcorr"])
    )
    print(
        "M/T/P corr gamma kernel -CIF w/bK vs corr gamma kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    plt.title("Gamma", fontsize=7)
    plt.xlabel("")
    plt.xticks([])
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=7)
    plt.ylim([-1, 1])
    plt.xlim(-0.3, 0.3)
    yl = plt.ylim()
    xl = plt.xlim()
    #plt.xlim([xl[0]-0.2, xl[1]])
    sns.despine(ax=ax, bottom=True)
    add_letter(plt.gcf(), gs[0,-2], 'C', y=1.15, x=-1.2)

    ax = plt.subplot(gs[0, -1])

    print('$$$$ - > Testing Gamma correlation with beh. kernel against alpha correlation with beh. kernel')
    gamma = rs.query('comparison=="AFcorr"').groupby('subject').correlation.mean()
    alpha = rs.query('comparison=="lAFcorr"').groupby('subject').correlation.mean()
    print('$$$$', ttest_rel(gamma, alpha))
    sns.stripplot(
        data=rs,
        x="comparison",
        y="correlation",
        order=["lAFcorr"],# "lAF-CIFcorr"],
        #palette=colors,
        palette='gray'
    )
    table_to_data_source_file(4, 'C', 'low_freq_correlation_V1_kernel', rs.query('comparison=="lAFcorr"').loc[:, ['subject', 'correlation']])
    plt.ylim(yl)
    plt.plot(
        [-0.2, 0 + 0.2],
        [rescorr.loc[:, "lAFcorr"].mean(), rescorr.loc[:, "lAFcorr"].mean()],
        "k", lw=2,
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(0, 0.9, '*', fontsize=12, verticalalignment='center')
        
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(1, 0.9, '*', fontsize=12, verticalalignment='center')        
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), np.arctanh(rescorr.loc[:, "lAFcorr"])
    )
    print(
        "M/T/P corr alpha kernel -CIF w/bK vs corr alpha kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    plt.title("Alpha", fontsize=7)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("")
    if label:        
        plt.xticks([])

    else:
        plt.xticks([0, 1, 2], ["", "", ""], rotation=45)
    plt.xlabel("")
    plt.xlim(-0.3, 0.3)
    sns.despine(ax=ax, bottom=True, left=True)

    return rescorr


def plot_per_sample_resp(ax, ssd, area, area_label, integration_slice, ylim=None, 
    hull=False, acc=True, sig=False):
    k = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD", area=area))
    ka = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD_acc_contrast", area=area))

    samples_k = {i: [] for i in range(10)}
    samples_ka = {i: [] for i in range(10)}
    # Compute significances
    for subject in range(1, 16):
        _k = dp.sps_2lineartime(
            dp.get_ssd_per_sample(ssd.query("subject==%i" % subject), "SSD", area=area)
        )
        _ka = dp.sps_2lineartime(
            dp.get_ssd_per_sample(
                ssd.query("subject==%i" % subject), "SSD_acc_contrast", area=area
            )
        )
        for sample in range(10):
            i = _k.loc[:, sample].to_frame()
            i.loc[:, "subject"] = subject
            i = i.set_index("subject", append=True)
            samples_k[sample].append(i)
            i = _ka.loc[:, sample].to_frame()
            i.loc[:, "subject"] = subject
            i = i.set_index("subject", append=True)
            samples_ka[sample].append(i)

    samples_k = {
        k: pd.pivot_table(
            pd.concat(v).dropna(), index="subject", columns="latency", values=k
        )
        for k, v in samples_k.items()
    }
    samples_ka = {
        k: pd.pivot_table(
            pd.concat(v).dropna(), index="subject", columns="latency", values=k
        )
        for k, v in samples_ka.items()
    }

    if sig:
        for sample in range(1, 10):
            # print(area, (samples_k[sample]-samples_ka[sample]).loc[:, 0.1*sample:].head())
            draw_sig(
                ax,
                (samples_k[sample] - samples_ka[sample]).loc[:, 0.1 * sample :],
                fdr=True,
                color="k",
            )

    if hull:
        # Compute per subject curves
        k_subject = ssd.groupby('subject').apply(lambda x: dp.sps_2lineartime(dp.get_ssd_per_sample(x, "SSD", area=area)).max(1))
        ka_subject = ssd.groupby('subject').apply(lambda x: dp.sps_2lineartime(dp.get_ssd_per_sample(x, "SSD_acc_contrast", area=area)).max(1))
        
        plotwerr(ka_subject, color=figure_6colors['AccDecode'], label="Decoding of\naccumulated contrast")
        plotwerr(k_subject, color=figure_6colors["DCD"], label="Decoding of\nsample contrast")
        table_to_data_source_file(3, 'B', 'hull_curves_%s_accumulated_contrast'%area, ka_subject)
        table_to_data_source_file(3, 'B', 'hull_curves_%s_sample_contrast'%area, k_subject)
        #plt.plot(ka.index, ka.max(1), color=figure_6colors['AccDecode'], label="Decoding of\naccumulated contrast")
        #plt.plot(k.index, k.max(1), color=figure_6colors["DCD"], label="Decoding of\nsample contrast")

        # Compute 'plateau t-tests'
        print('##### PLATEAU T-Tests: ', area)
        slopes_early = ka_subject.loc[:, 0.5:1].apply(lambda x: linregress(x.index, x.values)[0], 1)
        slopes_middle = ka_subject.loc[:, 1:1.15].apply(lambda x: linregress(x.index, x.values)[0], 1)
        slopes_end = ka_subject.loc[:, 1.15:1.4].apply(lambda x: linregress(x.index, x.values)[0], 1)
        #axvline(1, color='k')
        #axvline(1.2, color='k')
        #axvline(1.4, color='k')
        #axvline(0.5, color='k')
        print('EARLY against 0', slopes_early.mean(), ttest_1samp(slopes_early, 0))
        print('MIDDLE against 0', slopes_middle.mean(), ttest_1samp(slopes_middle, 0))
        print('LATE against 0', slopes_end.mean(), ttest_1samp(slopes_end, 0))

        print('EARLY vs Plateau', ttest_rel(slopes_early, slopes_middle))
        print('Plateau vs LATE', ttest_rel(slopes_middle, slopes_end))
        print('')
    else:
        #plt.plot(ka.index, ka, color=figure_6colors['AccDecode'], lw=0.1)
        cvals = array([0.71, 0.33, 0.53, 0.75, 0.59, 0.57, 0.55, 0.61, 0.45, 0.58])        
        
        norm=matplotlib.colors.Normalize(-5, 10)
        cm = matplotlib.cm.get_cmap('BuPu')
        colors = [cm(norm(10-i)) for i, c in enumerate(cvals)]
        colors = sns.color_palette('gray', n_colors=12)        
        for i in range(k.shape[1]):            
            plt.plot(k.index, k.loc[:, i], color=colors[i], lw=0.5)
        k.columns.name = 'sample'
        table_to_data_source_file(3, 'A', 'sample_decoding_%s'%area, k)
    

    plt.ylim(ylim)
    plt.legend(fontsize=7)
    yl = plt.ylim()
    plt.fill_between(
        [integration_slice.start, integration_slice.stop],
        [yl[0], yl[0]],
        [yl[1], yl[1]],
        facecolor="k",
        alpha=0.125,
        zorder=-1,
        edgecolor="none",
    )
    plt.title(area_label, fontsize=7)
    plt.xlabel("Time", fontsize=7)
    plt.ylabel("Correlation", fontsize=7)


@memory.cache()
def _get_palette(palette, brain=None, ogl=False, views=["par", "med"]):
    brain = dp.plot_brain_color_legend(
        palette, brain=brain, ogl=ogl, subject="fsaverage"
    )
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache()
def _get_lbl_annot_img(
    palette, low=0.4, high=0.6, views=[["lat"], ["med"]], colormap="RdBu_r", thresh=0
):
    print(colormap)
    brain, non_itms = dp.plot_brain_color_annotations(
        palette, low=low, high=high, alpha=1, colormap=colormap
    )
    if len(non_itms) > 0:
        import matplotlib

        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap(colormap)
        non_itms = {key: np.array(cm(norm(val))) for key, val in non_itms.items()}
        # palette[name] = cm(norm(value))
        brain = dp.plot_brain_color_legend(non_itms, brain=brain, subject="fsaverage")
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache()
def _get_img(palette, low=0.3, high=0.7, views=[["lat"], ["med"]]):
    brain, _ = dp.plot_brain_color_annotations(palette, low=low, high=high)
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache
def get_cl_decoding_peaks():
    ssd = dp.get_ssd_data(ogl=False, restrict=False)
    p = dp.extract_latency_peak_slope(ssd).test_slope.Pair
    lat = p.vfcPrimary.groupby("subject").mean().mean()
    return dp.extract_peak_slope(ssd, latency=lat).test_slope.Pair


@memory.cache
def get_ogl_decoding_peaks():
    ssd = dp.get_ssd_data(ogl=True, restrict=False)
    p = dp.extract_latency_peak_slope(ssd).test_slope.Pair
    lat = p.V1.groupby("subject").mean().mean()
    return dp.extract_peak_slope(ssd, latency=lat).test_slope.Pair


def make_stimulus(contrast, baseline=0.5, ringwidth=3 / 4):
    contrast = contrast / 2
    low = baseline - contrast
    high = baseline + contrast
    shift = 1
    ppd = 45
    sigma = 75
    cutoff = 5.5
    radius = 4 * ppd
    ringwidth = ppd * ringwidth
    inner_annulus = 1.5 * ppd
    X, Y = np.meshgrid(np.arange(1980), np.arange(1024))

    # /* Compute euclidean distance to center of our ring stim: */
    # float d = distance(pos, RC);
    d = ((X - 1980 / 2) ** 2 + (Y - 1024 / 2) ** 2) ** 0.5

    # /* If distance greater than maximum radius, discard this pixel: */
    # if (d > Radius + Cutoff * Sigma) discard;
    # if (d < Radius - Cutoff * Sigma) discard;
    # if (d < Annulus) discard;

    # float alpha = exp(-pow(d-Radius,2.)/pow(2.*Sigma,2.));
    alpha = np.exp(-(d - radius) ** 2 / (2 * sigma) ** 2)

    # /* Convert distance from units of pixels into units of ringwidths, apply shift offset: */
    # d = 0.5 * (1.0 + sin((d - Shift) / RingWidth * twopi));

    rws = 0.5 * (1.0 + np.sin((d - shift) / ringwidth * 2 * np.pi))

    # /* Mix the two colors stored in gl_Color and secondColor, using the slow
    # * sine-wave weight term in d as a mix weight between 0.0 and 1.0:
    # */
    # gl_FragColor = ((mix(firstColor, secondColor, d)-0.5) * alpha) + 0.5;
    rws = high * (1 - rws) + low * rws
    rws[d > (radius + cutoff * sigma)] = 0.5
    rws[d < (radius - cutoff * sigma)] = 0.5
    rws[d < inner_annulus] = 0.5

    return rws


def get_buildup_slopes(df, tmin=-0.25, dt=0.25):
    X = df.query(
        "epoch=='stimulus' & cluster=='JWG_M1' & contrast=='choice' & ~(hemi=='avg')"
    )

    times = X.columns.values
    times = times[tmin < times]
    # print(times)
    res = []
    for t in times:
        slopes = _get_slopes(X, [t, t + dt])
        k = [
            {"subject": i + 1, "time": t + dt, "dt": dt, "slope": s}
            for i, s in enumerate(slopes)
        ]
        res.extend(k)
    return pd.DataFrame(res)


def get_slopes(df, time):
    X = df.query(
        "epoch=='stimulus' & cluster=='JWG_M1' & contrast=='choice' & ~(hemi=='avg')"
    )
    return _get_slopes(X, time)


def _get_slopes(X, time):

    slopes = []
    inters = []
    for subject in range(1, 16):
        T = (
            pd.pivot_table(
                data=X.query("subject==%i & 10<freq & freq<40" % subject), index="freq"
            )
            .loc[:, time[0] : time[1]]
            .mean(0)
        )
        x, y = T.index.values, T.values
        s, i, _, _, _ = linregress(x, y)
        slopes.append(s)
        inters.append(i)
        # plot(x, y)
        # plot(x, x*s+i)

    return np.array(slopes)


def get_decoding_buildup(ssd, area="JWG_M1"):
    k = []
    for subject in range(1, 16):
        _ka = (
            dp.sps_2lineartime(
                dp.get_ssd_per_sample(
                    ssd.test_slope.Lateralized.query("subject==%i" % subject),
                    "SSD_acc_contrast",
                    area=area,
                )
            )
            .mean(1)
            .to_frame()
        )
        _ka.loc[:, "subject"] = subject
        k.append(_ka)
    k = pd.concat(k)
    return pd.pivot_table(k, index="subject", columns="latency", values=0)


def _decoding_buildup_slopes(X, dt=0.1):
    times = X.columns.values
    dtime = np.diff(times)[0]
    S = []
    for t in times[(times > (times.min() + dt)) & (times < (times.max() - dt))]:

        s = (
            X.loc[:, t - dt : t + dt]
            .T.apply(lambda x: linregress(np.arange(len(x)) * dtime, x)[0])
            .to_frame()
        )

        s.loc[:, "latency"] = t
        s.columns = ["slope", "latency"]
        S.append(s)
    S = pd.concat(S)
    return pd.pivot_table(S, index="subject", columns="latency", values="slope")


def _cp_corr():
    import pickle

    rescorr = []
    pairwise = []
    for freq in list(range(1, 10)) + list(range(10, 115, 5)):
        fname = (
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
        ccs, K, kernels_d, _ = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
        kernels_d.set_index(["cluster", "rmcif", "subject"], inplace=True)
        KK = np.stack(kernels_d.kernel)
        for cluster in [
            "vfcPrimary",
            "vfcEarly",
            "vfcV3ab",
            "vfcIPS01",
            "vfcIPS23",
            "JWG_aIPS",
            "JWG_IPS_PCeS",
            "JWG_M1",
        ]:
            kernels = pd.DataFrame(KK, index=kernels_d.index).query(
                'cluster=="%s"' % cluster
            )
            alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
            for i in range(15):
                rescorr.append(
                    {
                        "AFcorr": np.corrcoef(alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                        "sum": (alls.loc[i + 1] ** 2).sum(),
                        "slope": linregress(np.arange(10), alls.loc[i + 1])[0],
                        "subject": i + 1,
                        "freq": freq,
                        "cluster": cluster,
                    }
                )
            for cluster2 in [
                "vfcPrimary",
                "vfcEarly",
                "vfcV3ab",
                "vfcIPS01",
                "vfcIPS23",
                "JWG_aIPS",
                "JWG_IPS_PCeS",
                "JWG_M1",
            ]:
                kernels2 = pd.DataFrame(KK, index=kernels_d.index).query(
                    'cluster=="%s"' % cluster2
                )
                alls2 = pd.pivot_table(
                    data=kernels2.query("rmcif==False"), index="subject"
                )
                for i in range(15):
                    pairwise.append(
                        {
                            "corr": np.corrcoef(alls.loc[i + 1], alls2.loc[i + 1])[
                                0, 1
                            ],
                            "freq": freq,
                            "c1": cluster,
                            "c2": cluster2,
                        }
                    )
    rescorr = pd.DataFrame(rescorr)
    pairwise = pd.DataFrame(pairwise)
    return rescorr.set_index(["subject", "cluster", "freq"]), pairwise


def nr_figureS8():#pairs):
    fig=plt.figure(figsize=(7.5, 5.5))
    import pickle

    area_names = {
        "vfcPrimary":'V1',
        "vfcEarly":'V2-V4',
        "vfcV3ab":'V3A/B',
        "vfcIPS01":'IPS0/1',
        "vfcIPS23":'IPS2/3',        
        'vfcLO': "LO1/2",
        'vfcTO': "MT/MST",
        'vfcPHC': "PHC",         
        'vfcVO': "VO1/2",
        "JWG_aIPS":"aIPS",
        #"JWG_IPS_PCeS",
        #"JWG_M1",
    }

    fig = plt.gcf()
    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        "vfcEarly":[1-1, slice(1, 2+1)],
        "vfcVO":[4-1, slice(2, 3+1)],
        "vfcTO":[3-1, slice(2, 3+1)],
        "vfcLO":[2-1, slice(2, 3+1)],
        "vfcPHC":[5-1, slice(2, 3+1)],
        "vfcV3ab":[2-1, slice(0, 1+1)],
        "vfcIPS01":[3-1, slice(0, 1+1)],
        "vfcIPS23":[4-1, slice(0, 1+1)],  
        "JWG_aIPS":[5-1, slice(0, 1+1)],
        #"JWG_IPS_PCeS":[6, slice(1, 2+1)],
        #"JWG_M1":[7, slice(1, 2+1)], 
        #"6d":[8, slice(1, 2+1)]
    }
    areas = list(area_names.keys())
    k10 = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f10.results.pickle",
            "rb",
        )
    )
    kernels10 = k10["kernels"]
    k55 = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f55.results.pickle",
            "rb",
        )
    )

    kernels55 = k55["kernels"]
    kernels10.set_index(["cluster", "rmcif", "subject"], inplace=True)
    kernels55.set_index(["cluster", "rmcif", "subject"], inplace=True)
    colors = sns.color_palette('viridis', n_colors=2)
    color_alpha = 'xkcd:watermelon'
    color_gamma = 'xkcd:lightish green'
    results = []
    with mpl.rc_context(rc=rc):
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        gs_i = gs[0, :].subgridspec(4, 5, wspace=0.6, hspace=4)

        for i, cluster in enumerate(areas):
        
            #if ('M1' in cluster) or ('aIPS' in cluster) or ('IPS_P' in cluster):
            #    continue
            KK10 = np.stack(kernels10.kernel)
            kk10 = pd.DataFrame(KK10, index=kernels10.index).query(
                'cluster=="%s"' % cluster
            )
            alls10 = pd.pivot_table(data=kk10.query("rmcif==True"), index="subject")
            alls10_sum = alls10.copy()
            alls10_sum.loc[:, 'cluster'] = cluster
            alls10_sum.loc[:, 'frequency'] = 'alpha'
            results.append(alls10_sum)
            KK55 = np.stack(kernels55.kernel)
            kk55 = pd.DataFrame(KK55, index=kernels55.index).query(
                'cluster=="%s"' % cluster
            )
            alls55 = pd.pivot_table(data=kk55.query("rmcif==True"), index="subject")
            alls55_sum = alls55.copy()
            alls55_sum.loc[:, 'cluster'] = cluster
            alls55_sum.loc[:, 'frequency'] = 'gamma'
            results.append(alls55_sum)

            if cluster == "vfcPrimary":
                continue
            pos = position[cluster]                
            ax = plt.subplot(gs_i[pos[1], pos[0]])
            #ax= plt.subplot(gs_a[i // 2, np.mod(i, 2)])
            

            plotwerr(alls10, label="Residual kernel, 10Hz", color=color_alpha)#, linestyle=':')
            table_to_data_source_file('S8', 'A', cluster + '_low_freq', alls10)
            draw_sig(ax, alls10,y=-0.008, fdr=False, color=color_alpha)
            plotwerr(alls55, label="Residual kernel, 55Hz", color=color_gamma)
            table_to_data_source_file('S8', 'A', cluster + '_gamma', alls55)
            draw_sig(ax, alls55, y=-0.01, fdr=False, color=color_gamma)

            plt.title(area_names[cluster], fontsize=7)
            plt.ylim([-0.015, 0.035])
            plt.axhline(0, color='k', zorder=-1, lw=1)
            if cluster == "vfcEarly":
                plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [-0.01, 0, 0.01, 0.02, 0.03], fontsize=7)
                plt.ylabel('Choice predictive activity\n(AUC-0.5)', fontsize=7)
                plt.xticks([0, 4, 9], [1, 5, 10], fontsize=7)
                plt.xlabel('Sample', fontsize=7)
                plt.legend(fontsize=7, frameon=False, loc='center', bbox_to_anchor= (0.5, 1.4))
            else:
                plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [], fontsize=7)
                plt.xticks([0, 4, 9], [], fontsize=7)            
                
            for x in range(10):
                plt.axvline(x, color='k', lw=.5, zorder=-10, alpha=0.25)
            sns.despine(ax=ax)
        
        add_letter(fig, gs[0, 0], 'A', x=-0.1, y=1.05)
    results = pd.concat(results)
    results = results.set_index(['cluster', 'frequency'], append=True).mean(1).to_frame()
    """
    ax = plt.subplot(gs[1, 0])    
    for i, area in enumerate(["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23",  "JWG_aIPS"]):
        alpha = results.query('cluster=="%s" & frequency=="%s"'%(area, 'alpha'))
        gamma = results.query('cluster=="%s" & frequency=="%s"'%(area, 'gamma'))
        jit = np.random.randn(15)*0.05
        plt.scatter((jit+i)-0.2, alpha, color=color_alpha)
        plt.plot([i-0.35, i-0.05], [alpha.mean(), alpha.mean()], color='k')
        s=''
        if ttest_1samp(alpha, 0)[1] < 0.05:
            s='*'
        if ttest_1samp(alpha, 0)[1] < 0.01:
            s='**'
        if ttest_1samp(alpha, 0)[1] < 0.001:
            s='***'
        text(i-0.2, 0.05, s)
        plt.scatter((jit+i)+0.2, gamma, color=color_gamma)
        plt.plot([i+0.05, i+0.35], [gamma.mean(), gamma.mean()], color='k')
        s=''
        if ttest_1samp(gamma, 0)[1] < 0.05:
            s='*'
        if ttest_1samp(gamma, 0)[1] < 0.01:
            s='**'
        if ttest_1samp(gamma, 0)[1] < 0.001:
            s='***'
        text(i+0.2, 0.05, s)
    plt.ylabel('Mean choice predictive\nactivity (AUC-0.5)', fontsize=7)
    plt.axhline(0, color='k', ls=':', zorder=-1, alpha=0.5)
    plt.yticks(fontsize=6)
    sns.despine(ax=ax, bottom=True)
    xticks(np.arange(6), [area_names[a] for a in ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23",  "JWG_aIPS"]], fontsize=7)
    add_letter(fig, gs[1, 0], 'B', x=-0.1, y=1.15)
    """
    #plt.savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/S8_figure_S8.pdf"
    #)


def print_timings(timings=None):
    if timings is None:
        timings = pd.read_hdf("/Users/nwilming/u/conf_analysis/results/all_timing.hdf")
    for sub, t in timings.groupby("snum"):
        RD = (t.ref_offset_t - t.ref_onset_t) / 1200
        pp = "Ref: [{:1.2f}-{:1.2f}], {:1.2f}".format(RD.min(), RD.max(), RD.mean())
        ref_delay = (t.stim_onset_t - t.ref_offset_t) / 1200
        pp += "R-S: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            ref_delay.min(), ref_delay.max(), ref_delay.mean()
        )
        RT = (t.button_t - t.stim_offset_t) / 1200
        pp += " || S-RT: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            RT.min(), RT.max(), RT.mean()
        )
        FB = (t.meg_feedback_t - t.button_t) / 1200
        pp += " || RT-FB: [{:1.2f}-{:1.2f}], {:1.2f}; [{:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}]".format(
            FB.min(),
            FB.max(),
            FB.mean(),
            *np.percentile(FB.dropna(), [25, 50, 75, 100])
        )
        delay = []
        for (d, b), tt in t.groupby(["day", "block_num"]):
            delay.append(
                (
                    (
                        tt.ref_onset_t.iloc[10:-10].values
                        - tt.meg_feedback_t.iloc[9:-11].values
                    )
                    / 1200
                )
            )
        delay = np.concatenate(delay)

        pp += " || delay: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            np.nanmin(delay), np.nanmax(delay), np.nanmean(delay)
        )
        print("{:>2s}".format(str(sub)), pp)

    return timings


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    import matplotlib.colors as mcolors
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_auc_sanity(aucs=None, cluster='JWG_IPS_PCeS'):
    if aucs is None:
        aucs = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/auc_sanity_check.hdf')
    auc = (aucs.query('cluster=="%s"'%cluster)
        .groupby(['confidence', 'response'], sort=False)
        .mean())
    aucv = auc# np.triu(auc)
    print(aucv)
    #aucv[aucv==0] = np.nan
    #print(auc)
    #print(np.tril(auc, -1))
    pcolormesh([0, 0.25, 0.5, 0.75,1], [0, 0.25, 0.5, 0.75,1][::-1], aucv, 
        cmap='RdBu_r', vmin=.35, vmax=0.65) 
        #aspect='equal') #extent=[0,1,0,1]
    plot([0, 0.5], [1, 1], 'b', lw=4, label='Resp: Reference')
    plot([.5, 1], [1, 1], 'g', lw=4, label='Resp: Test stim.')

    dy=0.0225
    plot([0, 0.25], [1.+dy, 1.+dy], 'k', lw=4, label='High confidence')
    plot([0.25, 0.5], [1+dy, 1+dy], color=(0.5, 0.5, 0.5), lw=4, label='Low confidence')
    plot([0.5, 0.75], [1+dy, 1+dy], color=(0.5, 0.5, 0.5), lw=4)
    plot([0.75, 1], [1+dy, 1+dy], 'k', lw=4)
    #plot([0, 0.5], [0, 0], 'r', lw=4)
    #plot([.5, 1], [0, 0], 'g', lw=4)
    plot([1, 1], [0, 0.5],  'g', lw=4)
    plot([1, 1], [.5, 1], 'b', lw=4)

    dy=0.015
    plot([1.+dy, 1.+dy],[0, 0.25], 'k', lw=4)
    plot([1+dy, 1+dy], [0.25, 0.5],  color=(0.5, 0.5, 0.5), lw=4)
    plot([1+dy, 1+dy], [0.5, 0.75], color=(0.5, 0.5, 0.5), lw=4)
    plot([1+dy, 1+dy], [0.75, 1], 'k', lw=4)
    #plot([0, 0], [0, 0.5],  'g', lw=4)
    #plot([0, 0], [.5, 1], 'r', lw=4)
    ylim([-.1,1.1])
    xlim([-.1,1.1])
    xticks([])
    yticks([])
    sns.despine(left=True, bottom=True)


def plot_coupling(subject=None, latency=0.1, motor_lat=1.1):
    df = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/all_couplings_new_par.hdf')
    df = df.set_index(['motor_latency', 'readout_latency', 'subject']).stack().to_frame().reset_index()
    df.columns = ['motor_latency', 'readout_latency', 'subject', 'type', 'correlation']
    df.loc[:, 'readout_latency'] = np.around(df.readout_latency, 4)
    df.loc[:, 'motor_latency'] = np.around(df.motor_latency, 4)
    latencies = df.readout_latency.unique()
    latency = latencies[np.argmin(np.abs(latencies-latency))]
    print('Latency:', latency)

    latencies = df.motor_latency.unique()
    motor_lat = latencies[np.argmin(np.abs(latencies-motor_lat))]
    print('Motor Latency:', motor_lat)    

    if subject is not None:
        df = df.query('subject==%i'%subject)

    with mpl.rc_context(rc=rc):
        plt.figure(figsize=(10, 6))
        gs = matplotlib.gridspec.GridSpec(5, 2, width_ratios=[1, 0.5], 
            height_ratios=[1.5, 0.5, 0.8, 0.8, 0.8], )
        ax = subplot(gs[0,0])
        sns.lineplot(x='motor_latency', y='correlation', hue='type', 
            data=(
                df.query('readout_latency==%f'%latency)
                  .groupby(['motor_latency', 'type'])
                  .mean()
                  .reset_index())
        )
        plt.axvline(motor_lat, color='k', alpha=0.5,ls=':')
        plt.xlim([0,1.2])
        sns.despine()
        legend(loc=1, bbox_to_anchor=(1.6,1), frameon=False)
        for i, tpe in enumerate(['weighted_score', 'integrator_score', 'last_sample']):
            ax = subplot(gs[i+2,0])

            K = pd.pivot_table(df.query('type=="%s"'%tpe), 
                columns='readout_latency', 
                index='motor_latency', 
                values='correlation')
            ml = K.index.values
            rl = K.columns.values
            imshow(np.flipud(K.T), 
                extent=[ml.min(), ml.max(), rl.min(), rl.max()], 
                aspect='auto',
                cmap='RdBu_r',
                vmin=-0.1, 
                vmax=0.1)
            sns.despine(ax=ax, left=True, bottom=True)
            if i==2:
                xticks([0, 1])
                xlabel('Time of\nmotor readout')
            else:
                xticks([])
            yticks([])
            plt.arrow( 0, latency, 0.01, 0, fc="k", ec="k",
                head_width=0.025, head_length=0.025 )
            plt.xlim([0,1.2])
            plt.title(tpe, fontsize=7)
        sns.despine(ax=ax, left=False, bottom=False)
        yticks([0, 0.1, 0.2])
        ylabel('V1 readout')

        ax = subplot(gs[2:,1])
        sns.lineplot(x='readout_latency', y='correlation', hue='type', 
            data=(
                df.query('motor_latency==%f'%motor_lat)
                  .groupby(['readout_latency', 'type'])
                  .mean()
                  .reset_index())
        )
        #plt.xlim([0,1.2])
        plt.yticks([0, 0.05])
        plt.axvline(latency, color='k', alpha=0.5,ls=':')
        sns.despine(ax=ax, left=False, bottom=False)
        legend('', frameon=False)
    #plt.savefig('/Users/nwilming/Desktop/coupling_new.pdf')
    plt.figure(figsize=(10, 3))
    
    for sub, d in df.query('type=="integrator_score"').groupby('subject'):
        ax = subplot(4,4,sub)
        K = pd.pivot_table(d, 
            columns='readout_latency', 
            index='motor_latency', 
            values='correlation')
        ml = K.index.values
        rl = K.columns.values
        imshow(np.flipud(K.T), 
            extent=[ml.min(), ml.max(), rl.min(), rl.max()], 
            aspect='auto',
            cmap='RdBu_r',
            vmin=-0.15, 
            vmax=0.15)
        xticks([])
        yticks([])
        ylabel('S%i'%sub)
        sns.despine(ax=ax, left=True, bottom=True)
        #plt.savefig('/Users/nwilming/Desktop/coupling_new_ind_subs.pdf')


def nr_figure5(suffix='', motor_window=slice(-0.25, 0.25), vx_window=slice(-0.25, 0.25),
    average_peak=False, motor_area="JWG_M1", yl=[-0.25, 0.345]):
    fig = figure(figsize=(7.5, 4.5))    
    gs = matplotlib.gridspec.GridSpec(1,1)
    nr_plot_tl(gs_all=gs, motor_areas=[motor_area], yl=yl) # gs[0,:].subgridspec(1,1)
    
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/figure5%s.pdf'%suffix, bbox_inches='tight')
    #fig = figure()
    #gs = matplotlib.gridspec.GridSpec(2,1, height_ratios=[2, 1], hspace=0.5)
    #axa, axb, axc = nr_leakage_supp_figure(
    #    gs_all=gs[1,0], 
    #    motor_window=motor_window, 
    #    vx_window=vx_window,
    #    average_peak=average_peak, 
    #   motor_area=motor_area
    #)
    #sns.despine(ax=axa)
    #sns.despine(ax=axb)
    #sns.despine(ax=axc)
    #add_letter(fig, gs[0,:], 'A', x=-0.071, y=1.1)
    #add_letter(fig, axa, 'B', x=-0.4, y=1.1, new_axis=False)
    #add_letter(fig, axb, 'C', x=-0.4, y=1.1, new_axis=False)
    #add_letter(fig, axc, 'D', x=-0.4, y=1.1, new_axis=False)
    #sns.despine(ax=axb, bottom=False, left=False)

    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/figure5%s.pdf'%suffix, bbox_inches='tight')


def nr_leakage_supp_figure(gs_all=None, 
    motor_window=slice(-0., 0.25), vx_window=slice(-0.0, 0.25),
    average_peak=False,
    motor_area="JWG_M1"):
    # Plot M1 lateralized beta vs M1 averaged alpha
    leak = False
    areas = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",        
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "JWG_M1":"M1-hand",        
    }
    hemi = 'Averaged'
    fdr = False
    cluster_test = False
    leak_hemi_primary = 'Averaged'
    leak_hemi_motor = 'Lateralized'
    leak_freq_primary = 'alpha'
    leak_freq_motor = 'beta'
    yl = [-0.1, 0.25]
    tl = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_time_lagged_corrs_with_choice_decoder_time_window-0.1-0.8.hdf') #'/Users/nwilming/u/conf_analysis/results/nr_time_lagged_corrs.hdf')#_with_choice_decoder.hdf')

    #figure(figsize=(5.5, 2))
    #colors = {'JWG_M1':'black', 'vfcPrimary':'xkcd:lightish blue', 'vfcEarly':'xkcd:vermillion'}
    #colors['6d'] = (0, 0, 0)
    colors = _stream_palette()
    tl = tl.query('hemi=="%s"'%hemi)
    if gs_all is None:
        gs_all = matplotlib.gridspec.GridSpec(1, 1, hspace=0.3)[0,0]

    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        #"vfcEarly":[0, slice(1, 2+1)],
    
    }
    tl = tl.query('motor_area=="%s"'%motor_area)
    print(tl)
    scores_res = {}
    gs = gs_all.subgridspec(1, 3,  wspace=1)
    first = False
    for i, (motor_area, motor_data) in enumerate(tl.groupby('motor_area')): 
        m1x_motor = pd.pivot_table(columns='lag', values='corr', 
                        index='subject',                     
                        data=motor_data.query('freqband!="[45, 65]" & cluster=="%s"'%motor_area))                       
        
        
        motor_idxmax = [m1x_motor.loc[subject, motor_window].idxmax() for subject in range(1, 16)]
        if average_peak:
            motor_idxmax = [m1x_motor.loc[:, motor_window].mean(0).idxmax()]*16
            print('average peak for motor', motor_idxmax[0])
        for j, (freqband, freqdata) in enumerate(motor_data.groupby('freqband')):

            if "[45, 65]" == freqband:
                continue
            for (vfc, vfcdata) in freqdata.groupby('cluster'):
                
                if not vfc in areas:
                    continue
                
                
                m1x_primary = pd.pivot_table(columns='lag', values='corr', 
                    index='subject', 
                    data=vfcdata)                                
                # Define argmax in m1x_primary:
                scores_primary, scores_motor, idxmaxs = [], [], []
                for subject in range(1, 16):
                    idxmax = m1x_primary.loc[subject, vx_window].idxmax()
                    if average_peak:
                        idxmax = m1x_primary.loc[:, vx_window].mean(0).idxmax()

                    idxmaxs.append(idxmax)
                    scores_primary.append(m1x_primary.loc[subject, idxmax]) # - m1x_primary.loc[subject, 0]                    
                    scores_motor.append(m1x_motor.loc[subject, 0]) # m1x_motor.loc[subject, idxmax] - 
                #if vfc=="vfcIPS23":
                    #print(idxmaxs)
                    #import ipdb; ipdb.set_trace()
                scores_res[vfc] = (idxmaxs, scores_primary, scores_motor)                
                if average_peak:
                    print('average peak for', vfc, idxmaxs[0])
                if vfc in position:                    
                    pos = position[vfc]
                    ax = plt.subplot(gs[0    , pos[0]])
                    #plotwerr(m1x_primary, color=colors[vfc], label=areas[vfc])
                    plot(m1x_primary.mean(0), color=colors[vfc], label=areas[vfc])
                    plt.ylabel('Correlation', fontsize=7)
                    plt.xlabel('Lag [ms]', fontsize=7)
                    plt.axvline(np.mean(idxmaxs), color=colors[vfc], alpha=0.5, lw=0.5)
                    plt.axvline(0., color='k', alpha=0.5, lw=0.5)
                    #plt.axvline(-0.1, color='k', alpha=0.5, lw=0.5)
                    plt.xticks(fontsize=6)
                    plt.yticks(fontsize=6)
                    # Draw motor

                    
                    #if not first:                               
                    plot(m1x_motor.mean(0), color=colors['JWG_M1'], label='Reference\ncorrelation', zorder=-1)
                    #first=True
                    scatter(np.random.randn(15)*0.01, m1x_motor.loc[:, 0].values.ravel(), 3, 'k',  facecolors='none')

                
                #title(areas[vfc], fontsize=7)
                
                plt.ylim([-0.5, 0.8])
                
                if vfc == "vfcPrimary":
                #    plt.legend(fontsize=7)
                    plt.xlabel('      Lag [ms]      \n Vx leads                      M1 leads', fontsize=7)
                plt.xlim([-0.25, 0.25])
    axhline(0, color='gray', zorder=-1)
    plt.legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(.9, 0.85))
    #
    area_ticks = ['M1-hand']
    ax2= plt.subplot(gs[0, 1])
    for j, vfc in enumerate(['vfcPrimary', 'vfcEarly', "vfcV3ab", "vfcIPS01", "vfcIPS23"]):
        idxmax, scores_primary, scores_motor = scores_res[vfc]    
        #print('#######', len(scores_primary))
        if j == 0:
            for sm in scores_motor:
                jitter = np.random.randn()*0.05
                scatter([jitter], [sm], 5, facecolors='none', edgecolors=colors['JWG_M1'])
            plot([-0.3, 0.3], [np.mean(scores_motor), np.mean(scores_motor)], 'k')
        for sp in scores_primary:
            jitter = np.random.randn()*0.05            
            scatter([1+j+jitter], [sp], 5, facecolors='none', edgecolors=colors[vfc], zorder=1000)
        plot([-0.3+j+1, 0.3+1+j], [np.mean(scores_primary), np.mean(scores_primary)], color='k')
        #1/0
        #print(vfc,' IDXMAX mean:', np.mean(idxmax), 'STD:', np.std(idxmax), 'Min:', np.min(idxmax), 'Max:', np.max(idxmax))
        
        t, p = ttest_rel(scores_primary, scores_motor)    
        #print(vfc, np.mean(scores_primary), np.mean(scores_motor), ttest_rel(scores_primary, scores_motor))

        t2, p2 = ttest_1samp(scores_primary, 0) 
        #print(vfc, p2)
        s=''
        if p<0.05:
            s='*'
        if p<0.01:
            s='**'
        if p<0.001:
            s='***'
        text(1+j, 1, s, fontsize=7, horizontalalignment='center')
        ylim([-.5, 1.02])
        ylabel('Correlation', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        #text(0, -0.7-j*0.2, r'M1 vs. %s: t=%0.2f; p=%0.03f'%(areas[vfc], t, p), fontsize=7)
        print('M1 vs. %s: t=%0.5f; p=%0.06f'%(areas[vfc], t, p))

        #text(1+j, -.75, r'p=%0.3f'%np.round(p, 3), color='r', rotation=45, fontsize=6, horizontalalignment='center')
        area_ticks.append(areas[vfc])
    axhline(0, color='gray', zorder=-1)
    area_ticks[0] = 'M1-hand\n(lag 0)'
    xticks(np.arange(len(area_ticks)), area_ticks, fontsize=7, rotation=90)
    
    area_ticks = ['M1-hand']
    ax3= plt.subplot(gs[0, 2])
    for j, vfc in enumerate(['vfcPrimary', 'vfcEarly', "vfcV3ab", "vfcIPS01", "vfcIPS23"]):
        idxmax, scores_primary, scores_motor = scores_res[vfc]    
        
        if j == 0:
            for opo, sm in enumerate(motor_idxmax):
                jitter = np.random.randn()*0.075
                scatter([jitter], [sm], 5, facecolors='none', edgecolors=colors['JWG_M1'])
            plot([-0.3, 0.3], [np.mean(motor_idxmax), np.mean(motor_idxmax)], 'k')
        for sp in idxmax:
            jitter = np.random.randn()*0.075            
            scatter([1+j+jitter], [sp], 5, facecolors='none', edgecolors=colors[vfc])
        plot([-0.3+j+1, 0.3+1+j], [np.mean(idxmax), np.mean(idxmax)], color='k')
        #1/0
        #print(vfc,' IDXMAX mean:', np.mean(idxmax), 'STD:', np.std(idxmax), 'Min:', np.min(idxmax), 'Max:', np.max(idxmax))
        t, p = ttest_rel(idxmax, scores_motor)    
        #print(vfc, np.mean(idxmax), np.mean(motor_idxmax), ttest_rel(idxmax, motor_idxmax))

        t2, p2 = ttest_1samp(idxmax, 0) 
        #print(vfc, p2)
        s=''
        if p2<0.05:
            s='*'
        if p2<0.01:
            s='**'
        if p2<0.001:
            s='***'
        text(1+j, .25, s, fontsize=7, horizontalalignment='center')
        ylim([-.35, .35])
        ylabel('Lag at\nPeak corr.', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        #text(0, -0.7-j*0.2, r'M1 vs. %s: t=%0.2f; p=%0.03f'%(areas[vfc], t, p), fontsize=7)
        area_ticks.append(areas[vfc])
    axhline(0, color='gray', zorder=-1)


    xticks(np.arange(len(area_ticks)), area_ticks, fontsize=7, rotation=90)
    #sns.despine()
    #plt.tight_layout()
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_leakage_peak_trough_dist_mean_fixed.pdf')
    return ax, ax2, ax3


#def nr_figureS11():
#    nr_plot_tl()
#    plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS11.pdf')


def nr_plot_tl(tl=None, hemi="Averaged", suffix='acc_contrast',  leak=False, 
    leak_freq_primary='alpha', leak_freq_motor='alpha', 
    leak_hemi_primary='Averaged', leak_hemi_motor='Averaged', 
    fdr=False, cluster_test=False, yl=[-0.25, 0.275], gs_all=None,
    motor_areas=['JWG_M1', 'JWG_IPS_PCeS', '6d']):
    from matplotlib import gridspec
    areas = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",
        "vfcVO":"VO1/2",
        "vfcPHC":"PHC",
        "vfcTO":"MT/MST",
        "vfcLO":"LO1/2",
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "JWG_aIPS":"aIPS",
        "JWG_IPS_PCeS":"IPS/PostCeS",
        "JWG_M1":"M1-hand",
        "6d":'PMd',
    }

    if tl is None:
        tl = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_time_lagged_corrs_with_choice_decoder_time_window-0.1-0.8.hdf')#'/Users/nwilming/u/conf_analysis/results/nr_time_lagged_corrs_with_choice_decoder.hdf')
    if leak is False:
        leak = None
    else:
        leak = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_leakage_control.hdf') 

    colors = _stream_palette()
    colors['6d'] = (0, 0, 0)
    tl = tl.query('hemi=="%s"'%hemi)
    if gs_all is None:
        gs_all = gridspec.GridSpec(3, 1, hspace=0.5)
    
    #fig = figure(figsize=(7.5, 7))
    fig = plt.gcf()
    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        "vfcEarly":[1, slice(1, 2+1)],
        "vfcVO":[4, slice(2, 3+1)],
        "vfcTO":[3, slice(2, 3+1)],
        "vfcLO":[2, slice(2, 3+1)],
        "vfcPHC":[5, slice(2, 3+1)],
        "vfcV3ab":[2, slice(0, 1+1)],
        "vfcIPS01":[3, slice(0, 1+1)],
        "vfcIPS23":[4, slice(0, 1+1)],  
        "JWG_aIPS":[5, slice(0, 1+1)],
        #"JWG_IPS_PCeS":[6, slice(1, 2+1)],
        #"JWG_M1":[7, slice(1, 2+1)], 
        #"6d":[8, slice(1, 2+1)]
    }
    tl = tl.query('hemi=="Averaged"')
    #motor_areas = ['JWG_M1', 'JWG_IPS_PCeS', '6d']
    ltrs = ['A', 'B', 'C']
    for i, motor_area in enumerate(motor_areas):
        motor_data = tl.query('motor_area=="%s"'%motor_area)
        
        gs = gs_all[i, :].subgridspec(4, 6, wspace=0.6, hspace=4)
        m1hand_data = pd.pivot_table(columns='lag', values='corr', 
            index='subject', 
            data=motor_data.query('freqband=="[9, 11]" & cluster=="JWG_M1"'))
        
        for j, (freqband, freqdata) in enumerate(motor_data.groupby('freqband')):
            if "[45, 65]" == freqband:
                continue
            for vfc, vfcdata in freqdata.groupby('cluster'):
                
                if not vfc in areas:
                    continue

                if vfc not in position:
                    continue
                pos = position[vfc]                
                ax = plt.subplot(gs[pos[1], pos[0]])
                if vfc == "vfcPrimary":
                    m1hand_m = m1hand_data.loc[:, 0].mean()
                    m1hand_sem = m1hand_data.loc[:, 0].std()/(15**.5)
                    plot([0,], [m1hand_m], '.', color=colors['JWG_M1'], zorder=99999999+1)
                    plot([0, 0], [m1hand_m-m1hand_sem, m1hand_m+m1hand_sem], color=colors['JWG_M1'], zorder=99999999+1, alpha=0.75)
                    table_to_data_source_file(5, 'A', 'reference_correlation', m1hand_data.loc[:, 0].to_frame())
                m1x = pd.pivot_table(columns='lag', values='corr', 
                    index='subject', 
                    data=vfcdata)
                
                if freqband == "[45, 65]":
                    plotwerr(m1x, color=colors[vfc], linestyle=':', label='Gamma')
                else:
                    plotwerr(m1x, color=colors[vfc], label="Alpha", zorder=99999999)
                    table_to_data_source_file(5, 'A', vfc, m1x)
                #import ipdb; ipdb.set_trace()
                draw_sig(plt.gca(), m1x, fdr=fdr, cluster_test=cluster_test)
                draw_sig(plt.gca(), m1x.sub(m1x.loc[:, 0], 0), y=-0.025, color=colors[vfc], cluster_test=False, fdr=False)
                draw_sig(plt.gca(), m1x.sub(m1hand_data.loc[:, 0],0), y=-0.05, color='r', cluster_test=False, fdr=False)

                peak_lag = m1x.mean(0).idxmax()
                if vfc == 'vfcPrimary':
                    text(peak_lag+0.05, 0.275, 'Peak latency\n' + r'$t=%0.2fs$'%peak_lag, linespacing=1, backgroundcolor='w', 
                        fontsize=6, horizontalalignment='center', zorder=1100000)
                else:
                    text(peak_lag+0.075, -.15, r'$t=%0.2fs$'%peak_lag, fontsize=6, 
                        horizontalalignment='center', backgroundcolor='w', zorder=10111)
                axvline(peak_lag, color='k', lw=0.5, zorder=-1, alpha=0.5)
                if leak is not None:
                    cmd = 'cluster=="%s" & motor_area=="%s" & hemi=="%s" & motor_hemi=="%s" & freqband=="%s" & motor_freqband=="%s"'%(
                        vfc, motor_area, leak_hemi_primary, leak_hemi_motor, leak_freq_primary,
                        leak_freq_motor)
                    #print(cmd)
                    leak1x = pd.pivot_table(
                        columns='lag', values='corr', 
                        index='subject', 
                        data=leak.query(cmd))
                
                    if freqband == "[45, 65]":
                        plotwerr(leak1x, color=(0.5, 0.5, 0.5), linestyle=':', label='Gamma', zorder=-1)
                    else:
                        plotwerr(leak1x, color=(0.5, 0.5, 0.5), label="Alpha", zorder=-1)
                    draw_sig(plt.gca(), leak1x, fdr=fdr, cluster_test=cluster_test)

                title(areas[vfc], fontsize=7, color=colors[vfc])
                #if (pos[0] == 0) and (j==1):                
                #    plt.text(0.5, 1.2, 'Choice decoding in\n%s'%areas[motor_area], horizontalalignment='center',
                #        verticalalignment='bottom', transform=plt.gca().transAxes)
                plt.ylim(yl)
                if pos[0] == 0:
                    plt.ylabel('Correlation', fontsize=7)
                if pos[1].start > 0:
                    plt.xlabel('Lag [ms]', fontsize=7)
                #plt.axvline(0.1, color='k', alpha=0.5, lw=0.5)
                plt.axvline(0., color='k', alpha=0.5, lw=0.5)
                #plt.axvline(-0.1, color='k', alpha=0.5, lw=0.5)
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                if vfc == "vfcPrimary":
                #    plt.legend(fontsize=7)
                    plt.xlabel('      Lag [ms]      \nV1 leads    M1 leads', fontsize=7)
                sns.despine(ax=ax)
                plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        #add_letter(fig, gs_all[i, :], ltrs[i], x=-0.1, y=1.15, new_axis=True)
        
    plt.tight_layout()
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/time_lags_%s_%s.pdf'%(hemi, suffix))
            

def nr_plot_power_levels_and_correlations(CC):
    figure(figsize=(8.5, 3*3.5))
    gs = matplotlib.gridspec.GridSpec(4, 1, hspace=0.3, height_ratios=[2, 1, 2, 1])#, width_ratios=[3, 1], wspace=0.3)
    nr_plot_power_levels(gs_all=gs)

    nr_plot_power_val_correlations(CC, contrast='all', 
        gs=gs[1, 0].subgridspec(3,1, height_ratios=[1,2,1])[1,:])#, wspace=2, height_ratios=[5, 10, 5], )[1, :])
    nr_plot_power_val_correlations(CC, contrast='stimulus', 
        gs=gs[3, 0].subgridspec(3,1, height_ratios=[1,2,1])[1,:])#, wspace=2, height_ratios=[5, 10, 5], )[1, :])

    plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/nr_power_levels_w_correlation.pdf')


def nr_figureS2():
    from matplotlib import gridspec
    from conf_analysis.meg import srtfr
    fig = figure(figsize=(7.5, 3*3.5))
    gs_all = gridspec.GridSpec(2, 1, height_ratios=[0.75, 2])
    nr_plot_baseline_spectra(gs_all)
    
    gs = gs_all[1, 0].subgridspec(2, 1, hspace=0.5)
    nr_plot_power_levels(gs_all=gs)
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_new_supp_for_figure2.pdf')
    fig = figure()
    nr_plot_power_slopes()


def nr_plot_baseline_spectra(gs_all=None):
    areas = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",
        "vfcVO":"VO1/2",
        "vfcPHC":"PHC",
        "vfcTO":"MT/MST",
        "vfcLO":"LO1/2",
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "JWG_aIPS":"aIPS",
        "JWG_IPS_PCeS":"IPS/PostCeS",
        "JWG_M1":"M1-hand"
    }
    
    data = (
        pd.read_hdf('/Users/nwilming/u/conf_analysis/results/baseline_power_spectra.hdf')
        .dropna()        
        .mean(1)
        .to_frame()
    )
    data.columns=['power']
    if gs_all is None:
        gs_all = matplotlib.gridspec.GridSpec(2, 1, hspace=0.5)
    
    
    letters = ['A']

    gs = gs_all[0, 0].subgridspec(4, 8, wspace=1, hspace=2.5)

    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        "vfcEarly":[1, slice(1, 2+1)],
        "vfcVO":[4, slice(2, 3+1)],
        "vfcTO":[3, slice(2, 3+1)],
        "vfcLO":[2, slice(2, 3+1)],
        "vfcPHC":[5, slice(2, 3+1)],
        "vfcV3ab":[2, slice(0, 1+1)],
        "vfcIPS01":[3, slice(0, 1+1)],
        "vfcIPS23":[4, slice(0, 1+1)],  
        "JWG_aIPS":[5, slice(0, 1+1)],
        "JWG_IPS_PCeS":[6, slice(1, 2+1)],
        "JWG_M1":[7, slice(1, 2+1)]
    }
    
    area_colors = _stream_palette()

    for cluster, d in data.groupby(['cluster']):
        try:
            position[cluster]
        except KeyError:
            continue
        m1x = pd.pivot_table(index='subject', columns='freq', values='power', data=d)
        pos = position[cluster]    
        ax = plt.subplot(gs[pos[1], pos[0]])
        plotwerr(m1x.loc[:, 5:], color=area_colors[cluster])
        table_to_data_source_file('S2', 'A', cluster, m1x.loc[:, 5:])
        if cluster =="vfcPrimary":
            ax.set_ylabel(r'Power ($fT^2$)', fontsize=7)
            ax.set_xlabel('Frequency (Hz)', fontsize=7)
        ax.set_yscale('log')
        ax.set_xscale('log')        
        ax.set_ylim([10e-27, 10e-24])
        #plt.xticks(fontsize=6)

        #ax.set_xticks([1, 10, 100], ['1', '10', '100'])
        ax.set_xticks([ 10, 100])
        ax.set_xticklabels([ r'10', r'100'], fontsize=6) #tex code
        #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax.get_xaxis().set_tick_params(fontsize=6)
        plt.yticks(fontsize=6)
        sns.despine(ax=ax)
    add_letter(gcf(), gs_all[0,0], letters[0], x=-0.1, y=1.1, new_axis=True)
        #plt.tight_layout()


def nr_plot_power_levels(gs_all=None):
    areas = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",
        "vfcVO":"VO1/2",
        "vfcPHC":"PHC",
        "vfcTO":"MT/MST",
        "vfcLO":"LO1/2",
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "JWG_aIPS":"aIPS",
        "JWG_IPS_PCeS":"IPS/PostCeS",
        "JWG_M1":"M1-hand"
    }

    data_all = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_power_values.hdf')
    

    if gs_all is None:
        gs_all = matplotlib.gridspec.GridSpec(2, 1, hspace=0.5)
    bands = data_all.index.get_level_values('band').unique()
    colors = {
        band: color
        for band, color in zip(
            ['theta', 'alpha', 'beta', 'gamma', 'hf'], sns.color_palette("magma", n_colors=len(bands) + 1)
        )
    }
    letters = ['B', 'C']
    for i, contrast in enumerate(['all', 'stimulus']):
        data = data_all.query('contrast=="%s"'%contrast)

        gs = gs_all[i, 0].subgridspec(4, 8, wspace=1, hspace=2.5)

        position = {
            "vfcPrimary":[0, slice(1, 2+1)],
            "vfcEarly":[1, slice(1, 2+1)],
            "vfcVO":[4, slice(2, 3+1)],
            "vfcTO":[3, slice(2, 3+1)],
            "vfcLO":[2, slice(2, 3+1)],
            "vfcPHC":[5, slice(2, 3+1)],
            "vfcV3ab":[2, slice(0, 1+1)],
            "vfcIPS01":[3, slice(0, 1+1)],
            "vfcIPS23":[4, slice(0, 1+1)],  
            "JWG_aIPS":[5, slice(0, 1+1)],
            "JWG_IPS_PCeS":[6, slice(1, 2+1)],
            "JWG_M1":[7, slice(1, 2+1)]
        }
        band2label = {'alpha':'Alpha [8-12) Hz', 'theta':'Theta [4-8) Hz', 'beta': 'Beta [12-45) Hz',
            'gamma':'Gamma [45-65) Hz', 'hf':'High-frequency [65-120) Hz'}
        area_colors = _stream_palette()
        axins = None
        axins_twin = None
        ax_dict = {}

        data = pd.concat([
            data.query('band=="theta"'), 
            data.query('band=="alpha"'),
            data.query('band=="beta"'),
            data.query('band=="gamma"'),
            data.query('band=="hf"')])
        for band in ['theta', 'alpha', 'beta', 'gamma', 'hf']:
            db = data.query('band=="%s"'%band)
            for (cluster, contrast), d in db.groupby(['cluster', 'contrast']):
                try:
                    position[cluster]
                except KeyError:
                    continue
                m1x = pd.pivot_table(index='subject', data=d).loc[:, -0.25:1.25]
                pos = position[cluster]
                if (cluster, contrast) in ax_dict:
                    ax = ax_dict[(cluster, contrast)]
                else:
                    ax = plt.subplot(gs[pos[1], pos[0]])
                    ax_dict[(cluster, contrast)] = ax
                
                plotwerr(m1x, ax=ax, color=colors[band], label=band2label[band])
                table_to_data_source_file('S2', letters[i], cluster, m1x)
                ax.set_title(areas[cluster], fontsize=7, color=area_colors[cluster])
                if contrast=='all':
                    ax.set_ylim([-100, 100])
                else:
                    ax.set_ylim([-75, 75])
                if cluster == "vfcPrimary":
                    ax.set_ylabel('Power\n(% signal change)', fontsize=7)
                if not pos[1].start == 0:
                    ax.set_xlabel('Time [s]', fontsize=7)
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                #ax.set_yticklabels(ax.get_yticklabels(),fontdict={'fontsize':6})
                if (cluster=="JWG_M1") and (i==0):
                    ax.legend(loc="upper center", bbox_to_anchor=(-0.5,-1.15), fontsize=7, frameon=False)

                if cluster=="vfcPrimary":
                    if axins is None:        
                        from matplotlib.patches import ConnectionPatch            
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        import matplotlib.patches as patches
                        axins = inset_axes(ax, width=0.4, height=0.4,
                            bbox_to_anchor=(0, 2),
                            bbox_transform=ax.transAxes, loc=2, borderpad=0)
                        con = ConnectionPatch(xyA=(0, 0), xyB=(0, 55), 
                          coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, arrowstyle="-", color='k', zorder=-1, linewidth=0.5)
                        axins.add_artist(con)
                        con = ConnectionPatch(xyA=(1, 0), xyB=(1, 55), 
                          coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, arrowstyle="-", color='k', zorder=-1, linewidth=0.5)
                        axins.add_artist(con)
                        rect = patches.Rectangle((0, 0), 1, 55, linewidth=0.5,edgecolor='k',facecolor='none', zorder=-1)
                        ax.add_patch(rect)

                    if band == 'gamma':
                        print('plotting gamma into inset')
                        axins.plot(m1x.mean(0), color=colors[band])
                        axins.set_xlim(0, 1)
                        sns.despine(ax=axins)                    
                        axins.set_title('')
                        plt.xticks(fontsize=6)
                        plt.yticks(fontsize=6)
                        axins.set_xticks([])                    
                        if contrast=='all':
                            axins.set_ylim(0, 65)                                        
                            axins.set_yticklabels([0, 50], fontdict={'fontsize':6, 'color':colors[band]})
                        else:
                            axins.set_ylim(0, 45)
                            #axins.set_xticks([0, 40])                                        
                            axins.set_yticklabels([0, 40], fontdict={'fontsize':6, 'color':colors[band]})
                    elif band == 'hf':
                        if axins_twin is None:
                            axins_twin = axins.twinx()
                            plt.xticks(fontsize=6)
                            plt.yticks(fontsize=6)
                        print('plotting hf into inset')
                        axins_twin.plot(m1x.mean(), color=colors[band])
                        axins_twin.set_xlim([0, 1])
                        if contrast == 'all':
                            axins_twin.set_ylim([0, 15])
                            axins_twin.set_yticks([10])
                            axins_twin.set_yticklabels([10], fontdict={'fontsize':6, 'color':colors[band]})
                        if contrast == 'stimulus':
                            axins_twin.set_ylim([0, 15])
                            axins_twin.set_yticks([8])
                            axins_twin.set_yticklabels([8], fontdict={'fontsize':6, 'color':colors[band]})
                        #axins_twin.set_xticks([])                    
                        
                        axins_twin.set_title('')

                sns.despine(ax=ax)
        add_letter(gcf(), gs_all[i,0], letters[i], x=-0.1, y=1.1, new_axis=True)
        #plt.tight_layout()
        
    return gs_all
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/rev_power_vals.pdf')


def nr_plot_power_slopes(gs_all=None):
    #fig = figure(figsize=(7.5, 7.5))
    figure(figsize=(4.5, 1.5))
    areas = {
        "vfcPrimary":"V1",
        #"vfcEarly":"V2-V4",
        #"vfcVO":"VO1/2",
        #"vfcPHC":"PHC",
        #"vfcTO":"MT/MST",
        #"vfcLO":"LO1/2",
        #"vfcV3ab":"V3A/B",
        #"vfcIPS01":"IPS0/1",
        #"vfcIPS23":"IPS2/3",
        #"JWG_aIPS":"aIPS",
        #"JWG_IPS_PCeS":"IPS/PostCeS",
        #"JWG_M1":"M1-hand"
    }

    data_all = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_power_values.hdf')
    
    if gs_all is None:
        gs_all = matplotlib.gridspec.GridSpec(1, 2, wspace=0.5)
    bands = data_all.index.get_level_values('band').unique()
    colors = {
        band: color
        for band, color in zip(
            ['theta', 'alpha', 'beta', 'gamma', 'hf'], sns.color_palette("magma", n_colors=len(bands) + 1)
        )
    }
    letters = ['A', 'B']
    for i, contrast in enumerate(['all', 'stimulus']):
        data = data_all.query('contrast=="%s"'%contrast)
        slopes = data.groupby(['subject', 'band', 'cluster']).apply(lambda x: linregress(x.loc[:, 0.5:1].columns.values, x.loc[:, 0.5:1].values)[0])
        ps = slopes.groupby(['band', 'cluster']).apply(lambda x: ttest_1samp(x.values, 0)[1])
        ms =slopes.groupby(['band', 'cluster']).mean()
        #gs = gs_all[i, 0].subgridspec(4, 8, wspace=1, hspace=2.5)
        gs = gs_all[0, i]#.subgridspec(4, 8, wspace=1, hspace=2.5)

        position = {
            "vfcPrimary":[0, slice(1, 2+1)],
            "vfcEarly":[1, slice(1, 2+1)],
            "vfcVO":[4, slice(2, 3+1)],
            "vfcTO":[3, slice(2, 3+1)],
            "vfcLO":[2, slice(2, 3+1)],
            "vfcPHC":[5, slice(2, 3+1)],
            "vfcV3ab":[2, slice(0, 1+1)],
            "vfcIPS01":[3, slice(0, 1+1)],
            "vfcIPS23":[4, slice(0, 1+1)],  
            "JWG_aIPS":[5, slice(0, 1+1)],
            "JWG_IPS_PCeS":[6, slice(1, 2+1)],
            "JWG_M1":[7, slice(1, 2+1)]
        }
        band2label = {'alpha':'Alpha [8-12) Hz', 'theta':'Theta [4-8) Hz', 'beta': 'Beta [12-45) Hz',
            'gamma':'Gamma [45-65) Hz', 'hf':'High-frequency [65-120) Hz'}
        area_colors = _stream_palette()
        axins = None
        axins_twin = None
        ax_dict = {}

        for (cluster), d in slopes.groupby(['cluster']):

            try:
                areas[cluster]
            except KeyError:
                continue            
            pos = position[cluster]
            if (cluster, contrast) in ax_dict:
                ax = ax_dict[(cluster, contrast)]
            else:
                #ax = plt.subplot(gs[pos[1], pos[0]])
                ax = plt.subplot(gs)
                ax_dict[(cluster, contrast)] = ax
            
            sns.stripplot(x='band', y=0, 
                data=d.reset_index(), 
                order=['theta', 'alpha', 'beta', 'gamma', 'hf'], 
                palette=colors)
            if 'vfcPrimary' in cluster:
                dd = d.reset_index()
                dd.columns = ['subject', 'band', 'cluster', 'slope']
                #dd.loc[:, 'band'] = dd.band.values.astype('S')
                #dd['band'] = dd['band'].astype('|S')
                
                ddx = pd.pivot_table(data=dd, columns='band', index='subject')
                table_to_data_source_file('S2', {'all':'B', 'stimulus':'C'}[contrast], 'slope_estimate_V1', ddx)


            xticks([0, 1, 2, 3, 4], ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], fontsize=6, rotation=90)
            yticks(fontsize=6)
            xlabel('')
            ylabel('Slope of band-limited\npower (t=[0.5, 1] s', fontsize=7)
            axhline(0, color='k', zorder=100, alpha=0.5)
            if not (cluster == "vfcPrimary"):
                xticks([])
                #yticks([])
                ylabel('')
            #ax.set_title(areas[cluster], fontsize=7, color=area_colors[cluster])    
            if any(np.abs(ylim())>100):
                plt.ylim([-75, 75])
            for j, band in enumerate(['theta', 'alpha', 'beta', 'gamma', 'hf']):
                p = ps.loc[(band, cluster)]
                m = ms.loc[(band, cluster)]
                plot([j-0.2, j+0.2], [m, m], color='k', lw=2, zorder=100)
                plot([j-0.2, j+0.2], [m, m], color='gray', lw=3, zorder=99)
                #print(m)
                s = ''
                if p < 0.05:
                    s = '*'
                if p< 0.01:
                    s= '**'
                if p<0.001:
                    s='***'
                text(j, ylim()[1]-ylim()[1]*0.05, s, rotation=0, verticalalignment='center', horizontalalignment='center', fontsize=7)
            if contrast=='all':
                title('Stimulus vs. baseline', fontsize=7)
            else:
                title('Stimulus stronger vs. weaker', fontsize=7)
            sns.despine(ax=ax)
        #add_letter(gcf(), gs_all[i,0], letters[i], x=-0.1, y=1.1, new_axis=True)
        #plt.tight_layout()
    #plt.savefig(
    #    '/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_power_slope_figure.pdf',
    #    bbox_inches='tight')


@memory.cache()
def _get_correlation_between_contrast_decoding_and_kernel():
    from conf_analysis.meg import cort_kernel as ck
    ssd = dp.get_ssd_data(ogl=False, restrict=False)
    ssd = ssd.test_corr.Averaged.query(
        'signal=="SSD" & epoch=="stimulus"'
    )
    o = np.round(
        ssd.index.get_level_values("latency") + (ssd.index.get_level_values("sample") / 10),
        3,
    )
    o.name = "time"
    ssd = ssd.set_index(o, append=True)

    hull_curves = (ssd
            .groupby('subject')
            .apply(
                lambda x: pd.pivot_table(index='sample', columns='time', data=x).max(0)
                )
            ).vfcPrimary

    dk, confidence_kernel = ck.get_kernels()
    tp = np.arange(0, 1, 0.1) + 0.19
    htime = hull_curves.columns.values
    idx = np.array([np.argmin(np.abs(htime-t)) for t in tp])
    hull_curves = hull_curves.loc[:, htime[idx]]
    CCs = np.array([np.corrcoef(dk.loc[i,:].values, hull_curves.loc[i, :].values)[0,1] for i in range(1, 16)])
    return CCs


def nr_figureS4(first_sample=2, suffix=''):
    nr_figureS4and5(psychophysical_kernel=True, first_sample=first_sample, suffix=suffix)


def nr_figureS5(first_sample=0):
    nr_figureS4and5(first_sample=first_sample, psychophysical_kernel=False)


def nr_figureS4and5(psychophysical_kernel=True, first_sample=0, suffix=''):
    fnr = 'S5' if not psychophysical_kernel else 'S4'

    figure(figsize=(4.5, 1.5))
    colors = {
        band: color
        for band, color in zip(
            ['theta', 'alpha', 'beta', 'gamma', 'hf'], sns.color_palette("magma", n_colors=5 + 1)
        )
    }
    gs = matplotlib.gridspec.GridSpec(1, 2, wspace=0.5)
    CC = nr_power_val_correlations(first_sample=first_sample)
    #nr_plot_power_val_correlations(CC, contrasts=['all', 'stimulus'], 
    #    gs=gs)
    
    CCk = _get_correlation_between_contrast_decoding_and_kernel()
    if not psychophysical_kernel:
        ax= plt.subplot(gs[0,0])
        Cv1 = CC.query('contrast=="all" & cluster=="vfcPrimary"')
        sns.stripplot(x='band', hue='band', y='decoding_corr', data=Cv1, palette=colors, 
            order=['theta', 'alpha', 'beta', 'gamma', 'hf'], ax=ax)
        
        XCv1 = pd.pivot_table(data=Cv1, index='subject', columns='band', values='decoding_corr')
        table_to_data_source_file(fnr, "A", "Power_change_relative_to_baseline", XCv1)

        Cv1pd = pd.pivot_table(data=Cv1, index='subject', columns='band', values='decoding_corr')
        
        for i, band in enumerate(['theta', 'alpha', 'beta', 'gamma', 'hf']):
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'k', lw=2, zorder=100)
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'gray', lw=3, zorder=90)

        ts, ps = ttest_1samp(np.arctanh(Cv1pd.loc[:, ['theta', 'alpha', 'beta', 'gamma', 'hf']].values),0)
        print('Decoding corr, contrast=all')
        print(ts, ps)
        xtp = xticks()[0]
        for i in range(len(ps)):
            s = ''
            if ps[i] < 0.05:
                s= '*'
            if ps[i] < 0.01:
                s = '**'
            if ps[i] < 0.001:
                s ='***'
            text(xtp[i], 0.90, s, horizontalalignment='center')

        print('Compare correlation between contrast vs. psych. Kernel and decoding corr vs. freq band:')
        print('(positive t-values: correlation between kernel and contrast decoding is smaller')
        for band in ['theta', 'alpha', 'beta', 'gamma', 'hf']:
            print(band, ':', ttest_rel(np.arctanh(Cv1pd.loc[:, band]), np.arctanh(CCk)))
        plt.xticks([0, 1, 2, 3, 4], ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.legend([], frameon=False)
        sns.despine(ax=gca(), bottom=False)
        #plt.xticks([0, 1, 2, 3, 4], []) #, ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.xlabel('')
        plt.yticks(fontsize=6)
        plt.ylabel('Correlation with\ncontrast decoding', fontsize=7)
        #plt.title('V1', fontsize=7)
        plt.axhline(0, color='k', zorder=-1, alpha=0.5)
        plt.ylim([-1,1])
        title('Power change relative to baseline\n', fontsize=7)

    if psychophysical_kernel:    
        ax= plt.subplot(gs[0,0])
        Cv1 = CC.query('contrast=="all" & cluster=="vfcPrimary"')
        sns.stripplot(x='band', hue='band', y='kernel_corr', data=Cv1, palette=colors, 
            order=['theta', 'alpha', 'beta', 'gamma', 'hf'], ax=ax)
        Cv1pd = pd.pivot_table(data=Cv1, index='subject', columns='band', values='kernel_corr')
        
        XCv1 = pd.pivot_table(data=Cv1, index='subject', columns='band', values='kernel_corr')
        table_to_data_source_file(fnr, "A", "Power_change_relative_to_baseline", XCv1)

        for i, band in enumerate(['theta', 'alpha', 'beta', 'gamma', 'hf']):
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'k', lw=2, zorder=100)
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'gray', lw=3, zorder=90)

        ts, ps = ttest_1samp(np.arctanh(Cv1pd.loc[:, ['theta', 'alpha', 'beta', 'gamma', 'hf']].values),0)    
        print('Kernel corr, contrast=all')
        print(ts, ps)
        xtp = xticks()[0]
        for i in range(len(ps)):
            s = ''
            if ps[i] < 0.05:
                s= '*'
            if ps[i] < 0.01:
                s = '**'
            if ps[i] < 0.001:            
                s ='***'
            text(xtp[i], 0.9, s, horizontalalignment='center')
            
        plt.xticks([0, 1, 2, 3, 4], ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.legend([], frameon=False)
        sns.despine(ax=gca(), bottom=False)
        #plt.xticks([0, 1, 2, 3, 4], [])#, ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.xlabel('')
        plt.yticks(fontsize=6)
        plt.ylim([-1,1])
        plt.ylabel('Correlation with\npsychophysical kernel', fontsize=7)
        #plt.title('V1', fontsize=7)
        plt.axhline(0, color='k', zorder=-1, alpha=0.5)
        title('Power change relative to baseline\n', fontsize=7)

    if not psychophysical_kernel:
        ax= plt.subplot(gs[0,1])
        Cv1 = CC.query('contrast=="stimulus" & cluster=="vfcPrimary"')
        sns.stripplot(x='band', hue='band', y='decoding_corr', data=Cv1, palette=colors, 
            order=['theta', 'alpha', 'beta', 'gamma', 'hf'], ax=ax)
        Cv1pd = pd.pivot_table(data=Cv1, index='subject', columns='band', values='decoding_corr')        

        XCv1 = pd.pivot_table(data=Cv1, index='subject', columns='band', values='decoding_corr')
        table_to_data_source_file(fnr,  "B", "Stimulus_stronger_vs_weaker", XCv1)

        for i, band in enumerate(['theta', 'alpha', 'beta', 'gamma', 'hf']):
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'k', lw=2, zorder=100)
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'gray', lw=3, zorder=90)

        ts, ps = ttest_1samp(np.arctanh(Cv1pd.loc[:, ['theta', 'alpha', 'beta', 'gamma', 'hf']].values),0)    
        print('Decoding corr, contrast=stimulus')
        print(ts, ps)
        xtp = xticks()[0]
        for i in range(len(ps)):
            s = ''
            if ps[i] < 0.05:
                s= '*'
            if ps[i] < 0.01:
                s = '**'
            if ps[i] < 0.001:
                s ='***'
            text(xtp[i], 0.9, s, horizontalalignment='center')
        plt.legend([], frameon=False)
        sns.despine(ax=gca(), bottom=False)
        plt.xticks([0, 1, 2, 3, 4], ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.xlabel('')
        plt.yticks(fontsize=6)
        plt.ylabel('Correlation with\ncontrast decoding', fontsize=7)
        #plt.title('V1', fontsize=7)
        plt.axhline(0, color='k', zorder=-1, alpha=0.5)
        plt.ylim([-1,1])
        title('Stimulus stronger vs. weaker\n', fontsize=7)

    if psychophysical_kernel:
        ax= plt.subplot(gs[0,1])
        Cv1 = CC.query('contrast=="stimulus" & cluster=="vfcPrimary"')
        sns.stripplot(x='band', hue='band', y='kernel_corr', data=Cv1, palette=colors, 
            order=['theta', 'alpha', 'beta', 'gamma', 'hf'], ax=ax)
        
        XCv1 = pd.pivot_table(data=Cv1, index='subject', columns='band', values='kernel_corr')
        table_to_data_source_file(fnr,  "B", "Stimulus_stronger_vs_weaker", XCv1)

        Cv1pd = pd.pivot_table(data=Cv1, index='subject', columns='band', values='kernel_corr')
        
        for i, band in enumerate(['theta', 'alpha', 'beta', 'gamma', 'hf']):
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'k', lw=2, zorder=100)
            plot([i-0.15, i+0.15], [Cv1pd.loc[:, band].mean(), Cv1pd.loc[:, band].mean()], 'gray', lw=3, zorder=90)

        ts, ps = ttest_1samp(np.arctanh(Cv1pd.loc[:, ['theta', 'alpha', 'beta', 'gamma', 'hf']].values),0)    
        print('Kernel corr, contrast=stimulus')
        print(ts, ps)
        xtp = xticks()[0]
        for i in range(len(ps)):
            s = ''
            if ps[i] < 0.05:
                s= '*'
            if ps[i] < 0.01:
                s = '**'
            if ps[i] < 0.001:
                s ='***'
            text(xtp[i], 0.9, s, horizontalalignment='center')
        plt.legend([], frameon=False)
        sns.despine(ax=gca(), bottom=False)
        plt.xticks([0, 1, 2, 3, 4], ['Theta', 'Alpha', 'Beta', 'Gamma', 'High-freq.'], rotation=90, fontsize=7)
        plt.xlabel('')
        plt.yticks(fontsize=6)
        plt.ylim([-1,1])
        plt.ylabel('Correlation with\npsychophysical kernel', fontsize=7)
        #plt.title('V1', fontsize=7)
        plt.axhline(0, color='k', zorder=-1, alpha=0.5)
        title('Stimulus stronger vs. weaker\n', fontsize=7)
    if not psychophysical_kernel:
        savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS5.pdf', bbox_inches='tight')
    if psychophysical_kernel:
        savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS4%s.pdf'%suffix, bbox_inches='tight')
    
    
@memory.cache()                    
def nr_power_val_correlations(first_sample=1, ogl=False):
    from conf_analysis.meg import cort_kernel as ck
    data = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_power_values.hdf')    
    area_colors = _stream_palette()
    ssd = dp.get_ssd_data(ogl=ogl, restrict=False)
    ssd = ssd.test_corr.Averaged.query(
        'signal=="SSD" & epoch=="stimulus"'
    )
    o = np.round(
        ssd.index.get_level_values("latency") + (ssd.index.get_level_values("sample") / 10),
        3,
    )
    o.name = "time"
    ssd = ssd.set_index(o, append=True)
    
    hull_curves = (ssd
            .groupby('subject')
            .apply(
                lambda x: pd.pivot_table(index='sample', columns='time', data=x).max(0)
                )
            )

    decision_kernel, _ = ck.get_kernels()
    res = []
    for (contrast, cluster, band), d in data.groupby(['contrast', 'cluster', 'band']):
        # Compute decoding profile for this param combination
        decoding = hull_curves[cluster]
        d.columns = np.round(d.columns, 3)
        d = d.loc[:, decoding.columns]
        # Compute idx closest to 190ms shift
        idx = [np.argmin(np.abs(d.columns.values-t)) for t in np.arange(0, 1, 0.1)+0.19]

        dk = d.iloc[:, idx]
        start_time = first_sample*0.19
        end_time = 1.+0.19
        #print(d.loc[1, start_time:end_time].columns)
        for subject in range(1, 16):            
            v = np.corrcoef(decoding.loc[subject, start_time:end_time], d.loc[subject,start_time:end_time])[0,1]
            v2 = np.corrcoef(decision_kernel.loc[subject, first_sample:], dk.loc[subject].values[0, first_sample:])[0,1]
            #import ipdb; ipdb.set_trace()
            res.append({'subject':subject, 'contrast':contrast, 
                'cluster':cluster, 'band':band, 'decoding_corr':v, 'kernel_corr':v2})
    return pd.DataFrame(res)


def nr_plot_power_val_correlations(CC, contrasts=['all'], gs=None):
    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        "vfcEarly":[1, slice(1, 2+1)],
        "vfcVO":[4, slice(2, 3+1)],
        "vfcTO":[3, slice(2, 3+1)],
        "vfcLO":[2, slice(2, 3+1)],
        "vfcPHC":[5, slice(2, 3+1)],
        "vfcV3ab":[2, slice(0, 1+1)],
        "vfcIPS01":[3, slice(0, 1+1)],
        "vfcIPS23":[4, slice(0, 1+1)],  
        "JWG_aIPS":[5, slice(0, 1+1)],
        "JWG_IPS_PCeS":[6, slice(1, 2+1)],
        "JWG_M1":[7, slice(1, 2+1)]
    }
    area_labels = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",
        "vfcVO":"VO1/2",
        "vfcPHC":"PHC",
        "vfcTO":"MT/MST",
        "vfcLO":"LO1/2",
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "JWG_aIPS":"aIPS",
        "JWG_IPS_PCeS":"IPS/P...",
        "JWG_M1":"M1-hand"
    }
    colors = {
        band: color
        for band, color in zip(
            ['theta', 'alpha', 'beta', 'gamma', 'hf'], sns.color_palette("magma", n_colors=5 + 1)
        )
    }
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(4,3)    
    areas = position.keys()
    
    for j, measure in enumerate(['decoding_corr', 'kernel_corr']): 
        for i, contrast in enumerate(contrasts):
            dcontrast = CC.query('contrast=="%s"'%contrast)
            ax = subplot(gs[i, 2*j+1])
            for band, dband in dcontrast.groupby('band'):
                D = pd.pivot_table(
                        index='subject', columns='cluster', values=measure, data=dband).loc[:, areas]
                D = D.rename(columns=area_labels)
                plotwerr(
                    D, color=colors[band]                    
                )
            plt.ylim([-1, 1])
            
            sns.despine(ax=plt.gca(), offset=10)
            #if measure == 'decoding_corr':
            #   plt.ylabel('Correlation with\ncontrast decoding', fontsize=7)
            #else:
            #    plt.ylabel('Correlation with\npsychophysical kernel', fontsize=7)
            plt.yticks(fontsize=6)
            sns.despine(ax=ax, left=True)
            plt.yticks([])
            plt.axhline(0, color='k', zorder=-1, alpha=0.5)
        if i==1:
            plt.xticks(fontsize=7, rotation=90)
        else:
            plt.xticks([])


def nr_figureS3():
    return nr_power_levels_by_var()

def nr_power_levels_by_var(band='gamma', yl=[20, 70]):
    from matplotlib import gridspec
    from scipy.stats import linregress
    import pingouin as pg
    def drophalf(x):
        xend = x.iloc[:, -1].values[0]
        xstart = x.iloc[:, 0].values[0]
        xmid = (xend-xstart)/2
        return x.columns.values[np.argmin(np.abs(x.values.ravel()-xmid))]-x.columns.values[0]

    fig = figure(figsize=(5.5, 5))
    gs = matplotlib.gridspec.GridSpec(2,1, hspace=0.5)#, width_ratios=[1, 0.5])
    ax = subplot(gs[0, 0])
    B = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/nr_power_levels_all_by_var.hdf')
    B.columns = B.columns.get_level_values('time')

    selection_slice=(0.266 <= B.columns.values) & (B.columns.values<=0.9)
    slopes = (
        B
            .loc[:, selection_slice]
            .groupby(['cluster', 'subject', 'band', 'variance'])
            .apply(lambda x:linregress(x.columns.get_level_values('time'), x.values.ravel())[0])
            .reset_index()
    )
    for dc, d in slopes.query('band=="%s"'%band).groupby(['cluster']):
        X = pd.pivot_table(data=d, index='subject', columns='variance', values=0)
        table_to_data_source_file('S3', 'AB', dc + '_slopes_gamma_band_power',X)
    hds = (
        B
            .loc[:, selection_slice]
            .groupby(['cluster', 'subject', 'band', 'variance'])
            .apply(drophalf).reset_index()
        )
    slopes = slopes.rename(columns={0:'slope'})
    slopes.loc[:, 'hdt'] = hds.loc[:, 0]

    b = B.query('cluster=="vfcPrimary" & band=="%s"'%(band)).stack().reset_index()
    for (dc, dv), d in B.query('band=="%s"'%band).groupby(['cluster', 'variance']):
        X = pd.pivot_table(data=d, index='subject',)
        table_to_data_source_file('S3', 'AB', dc + '_' 'Var' +dv + '_power_gamma_band_power',X)

    table_to_data_source_file('S3', 'AB', 'gamma_band_power', B.query('band=="%s"'%band))

    
    b.columns = ['subject', 'band', 'cluster', 'variance', 'time', 'power']
    colors = {'low':'#ff0066', 'mid':'#6666ff', 'high':'#009933'}
    variances = {'low':'Low', 'mid':'Medium', 'high':'High'}

    for var in ['low', 'mid', 'high']:
        dv = b.query('variance=="%s"'%var)
        s = pd.pivot_table(data=dv, index='subject', columns='time', values='power')        
        plot(s.loc[:, 0:1.2].mean(0), alpha=0.5, color=colors[var])
        selection = s.loc[:, selection_slice].mean(0)
        s,i, _, _, _ = linregress(selection.index.values, selection.values)
        plot(selection.index.values, selection.index.values*s+i, color=colors[var], label='%s variance'%variances[var])
    legend(frameon=False,fontsize=7)
    aov = pg.rm_anova(dv='slope', within=['variance'], subject='subject', 
        data=slopes.query('cluster=="vfcPrimary" & band=="%s"'%band), detailed=False)
    print('SLOPE: RM AONOVA W/O band as factor:')
    print(aov)

    aov = pg.rm_anova(dv='hdt', within=['variance'], subject='subject', 
        data=slopes.query('cluster=="vfcPrimary" & band=="%s"'%band), detailed=False)
    print('HDT: RM AONOVA W/O band as factor:')
    print(aov)

    print('')
    print('')
    aov = pg.rm_anova(dv='slope', within=['variance', 'band'], subject='subject', 
        data=slopes.query('cluster=="vfcPrimary"'), detailed=False)
    print('RM AONOVA WITH band as factor:')
    print(aov)
    sns.despine()
    xlabel('Time [s]', fontsize=7)
    xlim([0.15, 1])
    ylim(yl)
    xticks(fontsize=6)
    yticks(fontsize=6)
    ylabel('V1 Gamma band power\n(% signal change)', fontsize=7)

    add_letter(fig, gs[0,0], 'A', x=-0.1, y=1.1)
    nr_slopes_subplot(slopes, gs=gs[1,0])
    add_letter(fig, gs[1,0], 'B', x=-0.1, y=1.1)

    #_=xticks(np.arange(len(xticks_lbls)), xticks_lbls, rotation=90)
    #savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS3.pdf')
    return slopes


def nr_slopes_subplot(slopes, gs=None):
    from matplotlib import gridspec
    palette = _stream_palette()
    import pingouin as pg
    areas = {
        "vfcPrimary":"V1",
        "vfcEarly":"V2-V4",
        "vfcV3ab":"V3A/B",
        "vfcIPS01":"IPS0/1",
        "vfcIPS23":"IPS2/3",
        "vfcLO":"LO1/2",
        "vfcTO":"MT/MST",
        "vfcVO":"VO1/2",
        "vfcPHC":"PHC",        
        "JWG_aIPS":"aIPS",
        #"JWG_IPS_PCeS":"IPS/PostCeS",
        #"JWG_M1":"M1-hand"
    }
    xticks_lbls = []

    #gs = gridspec.GridSpec(1,2)
    position = {
        "vfcPrimary":[0, slice(1, 2+1)],
        "vfcEarly":[1, slice(1, 2+1)],
        "vfcVO":[4, slice(2, 3+1)],
        "vfcTO":[3, slice(2, 3+1)],
        "vfcLO":[2, slice(2, 3+1)],
        "vfcPHC":[5, slice(2, 3+1)],
        "vfcV3ab":[2, slice(0, 1+1)],
        "vfcIPS01":[3, slice(0, 1+1)],
        "vfcIPS23":[4, slice(0, 1+1)],  
        "JWG_aIPS":[5, slice(0, 1+1)],
        #"JWG_IPS_PCeS":[6, slice(1, 2+1)],
        #"JWG_M1":[7, slice(1, 2+1)], 
        #"6d":[8, slice(1, 2+1)]
    }
    
    gs_n = gs.subgridspec(4, 6, wspace=0.6, hspace=2.5)

    for i, cluster in enumerate(areas.keys()):
        pos = position[cluster]
        ax = plt.subplot(gs_n[pos[1], pos[0]])

        c_slopes = slopes.query('band=="gamma" & cluster=="%s"'%cluster)
        aov = pg.rm_anova(dv='slope', within=['variance'], subject='subject', 
            data=c_slopes, detailed=False)
        p = aov.loc[0, 'p-unc']

        X = pd.pivot_table(values='slope', index='subject', columns='variance', data=c_slopes).loc[:, ['low', 'mid', 'high']]
        xticks_lbls.append(areas[cluster])
        mX, sX = X.mean(), X.std()
        sem = sX/(X.shape[0]**.5)
        xs = np.array([-0.25, 0, 0.25])
        plot(xs, mX, color=palette[cluster])
        for jj in range(3):
            plot([xs[jj], xs[jj]], [mX[jj]-sX[jj], mX[jj]+sX[jj]], color=palette[cluster])
        ylim([-75, 10])
        xlim([-0.5, 0.5])
        if i == 0:
            xticks([-0.25, 0, 0.25], ['Low', 'Medium', 'High'], fontsize=6, rotation=90)
            xlabel('Variance', fontsize=7)
        else:
            xticks([-0.25, 0, 0.25], ['', '', ''])

        sigtxt = ''
        if p < 0.05:
            sigtxt = '*'
        if p < 0.01:
            sigtxt = '**'
        if p < 0.001:
            sigtxt = '***'
        #print(p, sigtxt)
        text(0, 0, sigtxt, fontsize=7, horizontalalignment='center')
        sns.despine(ax=ax, bottom=True)
        yticks([-50, -25, 0], fontsize=6)
        if i == 0:
            ylabel('Slope', fontsize=7)    
        #else:
        #    yticks([-50, -25, 0], ['', '', ''])
        title(areas[cluster], fontsize=7, color=palette[cluster])


def nr_figureS10(sub=3, dcd=None):
    if dcd is None:
        dcd = dp.get_decoding_data()
    dcd = pd.pivot_table(data=dcd.test_roc_auc.Lateralized.query('signal=="MIDC_split" & epoch=="stimulus"'), values="JWG_M1",  index='subject', columns='latency')
    dcd = dcd.loc[:, -0.4:1.4]-0.5
    kernel = pd.read_hdf("/Users/nwilming/u/conf_analysis/results/nr_high_res_kernels.h5")
    from matplotlib import gridspec
    palette = _stream_palette()
    fig=figure(figsize=(5.5, 2))
    gs = gridspec.GridSpec(1,2, wspace=0.3)
    ax = subplot(gs[0, 0])
    kernel = kernel.query('hemi=="Averaged" & cluster=="vfcPrimary" & freqband=="[9, 11]"').loc[:,-0.4:1.4]
    kernel = pd.pivot_table(index='subject',  data=kernel)
    plotwerr(kernel, color='xkcd:bright blue')
    table_to_data_source_file('S10', 'A', 'high_res_V1_kernel', kernel)
    #plotwerr(dcd)
    ylabel('AUC-0.5', fontsize=7)
    xlabel('Time [s]', fontsize=7)
    sns.despine(ax=ax)
    xticks(fontsize=6)
    yticks(fontsize=6)
    add_letter(fig, ax, 'A', x=-0.2, y=1.1, new_axis=False)
    """
    ax = subplot(gs[0, 1])


    plot(kernel.loc[sub,:].index.values, kernel.loc[sub,:].values, color='xkcd:light blue')
    plot(kernel.loc[sub,0.2:1], color='xkcd:bright blue', lw=2, label='V1 Alpha band kernel')
    plot(dcd.loc[sub, :].index.values, dcd.loc[sub, :].values, color='xkcd:tangerine')
    plot(dcd.loc[sub, -0.:0.8], color='xkcd:red orange', label='M1-hand choice decoding')
    xticks(fontsize=6)
    yticks(fontsize=6)

    corr2 = np.corrcoef(kernel.loc[sub,.2:1], dcd.loc[sub, 0:0.8])[0,1]
    corr = _compute_time_lagged_corr(dcd.loc[sub, -0.2:1.4], kernel.loc[sub, -0.2:1.4])
    corr = corr.loc[corr.lag==0.1980, 'corr'].values
    print(corr, corr2)
    ylim([-.1, .15])
    xlabel('Time [s]', fontsize=7)
    text(0, 0.14, r'$r=%0.2f$'%np.round(corr, 2), fontsize=7)
    sns.despine(ax=ax)
    legend(frameon=False, loc='lower right', fontsize=7)
    add_letter(fig, ax, 'B', x=-0.2, y=1.1, new_axis=False)
    #return corr
    """
    #savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS10.pdf', bbox_inches='tight')


def _compute_time_lagged_corr(signal_a, signal_b, tlow=0, thigh=0.8):
    """
    Compute temporal cross correlation.

    """
    signal_a = signal_a.loc[-0.2:1.4]
    signal_b = signal_b.loc[-0.2:1.4]
    assert len(signal_a) == len(signal_b)
    assert (np.isnan(signal_b.values).sum()==0)
    # Shift by +- 300ms
    signal_ref = signal_a.loc[tlow:thigh]
    # Average spacing between motor signals: 0.017 or 0.016
    idx = (tlow <= signal_b.index.values) & (signal_b.index.values <= thigh)
    results = []
    for lag in np.arange(-13, 14):
        c = np.corrcoef(signal_ref.values, signal_b.loc[np.roll(idx, lag)].values)[
            0, 1
        ]
        results.append({"lag": lag * 0.0165, "corr": c})
    return pd.DataFrame(results)


def nr_figureS11():
    nr_overall_delay_plot()


def nr_overall_delay_plot():
    from matplotlib import gridspec
    fig = figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(4, 4, hspace=0.75, wspace=0.75)

    ga = gs[0,:].subgridspec(1, 4, hspace=0.5, wspace=0.5)
    nr_delay_plot(gs=ga, time_window_ref=0.1, time_window_comp=0.1, one_cycle=False)
    add_letter(fig, ga[:, :], 'A', x=-0.1, y=1.1, new_axis=True)

    ga = gs[1,:2].subgridspec(1, 2, hspace=0.5, wspace=0.5)
    nr_delay_plot(gs=ga, plot_all=False, time_window_ref=0.1, time_window_comp=0.1, one_cycle=True)
    add_letter(fig, ga[:, :], 'B', x=-0.1, y=1.1, new_axis=True)

    ga = gs[1,2:].subgridspec(1, 2, hspace=0.5, wspace=0.5)
    nr_delay_plot(gs=ga, plot_all=False, time_window_ref=0.1, time_window_comp=0.1, one_cycle=False, ramp=True)
    add_letter(fig, ga[:, :], 'C', x=-0.1, y=1.1, new_axis=True)

    ga = gs[2,:2].subgridspec(1, 2, hspace=0.5, wspace=0.5)
    nr_delay_plot(gs=ga, plot_all=False, time_window_ref=0.1, time_window_comp=0.1, one_cycle=True, ramp=False, noise=True)
    add_letter(fig, ga[:, :], 'D', x=-0.1, y=1.1, new_axis=True)

    ga = gs[2,2:].subgridspec(1, 2, hspace=0.5, wspace=0.5)
    nr_delay_plot(gs=gs[2,2:].subgridspec(1, 2, hspace=0.5, wspace=0.5), plot_all=False, time_window_ref=0.1, time_window_comp=0.1, one_cycle=False, ramp=True, noise=True)
    add_letter(fig, ga[:, :], 'E', x=-0.1, y=1.1, new_axis=True)

    savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/delay_simulation_figure_mt_PA100ms.pdf')
    savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/delay_simulation_figure_mt_PA100ms.svg')


def nr_delay_plot(gs=None, 
        plot_all=True,
        time_window_ref=0.2, 
        time_window_comp=0.2, 
        one_cycle=True, 
        ramp=False, 
        comp=50, 
        noise=False,
        suffix='',
        panel=None
        ):    
    from matplotlib import gridspec
    shift, t, data_ref, data_comp, power_ref, power_comp, lags, cc = nr_delay_sim(
        time_window_comp=time_window_comp, 
        time_window_ref=time_window_ref, 
        one_cycle=one_cycle,
        comp=comp,
        ramp=ramp,
        noise=noise)
    if gs is None:
        if plot_all:
            gs = gridspec.GridSpec(1, 4, wspace=0.4)
        else:
            gs = gridspec.GridSpec(1, 2, wspace=0.4)
    #figure(figsize=(8, 8))
    ax=subplot(gs[0, 0])
    plt.plot(t, data_ref, label='10 Hz Signal', zorder=100)
    plt.plot(t, data_comp, label='%i Hz Signal'%comp)
    xlabel('Time (s)', fontsize=7)
    ylabel('Signal', fontsize=7)
    xlim([-0.2, 0.2])
    ylim([-1.2, 1.2])
    axvline(0, color='k', ls=':', zorder=-1)
    xticks(fontsize=6)
    yticks(fontsize=6)
    sns.despine(ax=ax)

    if plot_all:
        ax=subplot(gs[0, 1])
        plt.plot(t, power_ref, label='10 Hz Power\ntime window=%0.2f s'%time_window_ref, zorder=100)
        plt.plot(t, power_comp, label='%i Hz Power\ntime window=%0.2f s'%(comp, time_window_comp))
        xlim([-time_window_ref*4, 4*time_window_ref])
        xlabel('Time (s)', fontsize=7)
        ylabel('Normalized power', fontsize=7)
        #legend(frameon=False, loc='upper left', fontsize=7)#, bbox_to_anchor=(1, 0.9))
        xticks(fontsize=6)
        yticks(fontsize=6)
        sns.despine(ax=ax)
        ax=subplot(gs[0, 2])
        lags = np.array(lags)*1000
        plt.plot(lags, cc, color='k')
        axvline(shift*1000, color='xkcd:pink', ls=':')
        #xlim([-0.1, 0.1])
        xlabel('Lag (ms)\n10 Hz leads           50Hz leads', horizontalalignment='center', fontsize=7)
        ylabel('Cross-correlation\n(Pearson)', fontsize=7)
        xticks(fontsize=6)
        yticks(fontsize=6)
        sns.despine(ax=ax)
    f, l, ls = nr_delay_sim_all_f(one_cycle=one_cycle, 
        time_window_ref=time_window_ref, 
        time_window_comp=time_window_comp,
        ramp=ramp,
        noise=noise
    )

    if time_window_ref==0.1:
        f2, l2, ls2 = nr_delay_sim_all_f(one_cycle=one_cycle, 
            time_window_ref=time_window_ref, 
            time_window_comp=0.25,
            ramp=ramp,
            noise=noise
        )
    else:
        f2, l2, ls2 = nr_delay_sim_all_f(one_cycle=one_cycle, 
            time_window_ref=time_window_ref, 
            time_window_comp=time_window_comp*2,
            ramp=ramp,
            noise=noise
        )

    if plot_all:
        ax=subplot(gs[0, 3])
    else:
        ax=subplot(gs[0, 1])
    plt.plot(f[f>=10], l[f>=10]*1000, color='xkcd:pink')
    plt.plot(f2[f2<10], l2[f2<10]*1000, color='xkcd:light blue')
    if noise:
        plt.fill_between(f[f>=10], (l-ls)[f>=10]*1000, (l+ls)[f>=10]*1000, alpha=0.5, color='xkcd:pink')
        plt.fill_between(f2[f2<10], (l2-ls2)[f2<10]*1000, (l2+ls2)[f2<10]*1000, alpha=0.5, color='xkcd:light blue')
    plt.axhline(0, color='gray', ls=':')
    plt.axvline(10, color='gray', ls=':')
    plt.xlabel('Comparison\nFrequency (Hz)', fontsize=7)
    plt.ylabel('Peak Lag (ms)', fontsize=7)
    plt.xticks([10, 50, 100], fontsize=6)
    #plt.axhline(0.04*1000, color='gray', ls=':')
    plt.axvline(50, color='gray', ls=':')
    xticks(fontsize=6)
    yticks(fontsize=6)
    sns.despine(ax=ax)
    plt.ylim([-75, 75])
    #plt.tight_layout()
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_delay_sim%s.pdf'%suffix, bbox_inches='tight')
    #plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_delay_sim%s.svf'%suffix, bbox_inches='tight')


@memory.cache()
def nr_delay_sim_all_f(one_cycle=True, time_window_ref=0.1, time_window_comp=0.1, ramp=False, noise=False):
    F = np.arange(1/time_window_comp, 101, 1)
    if not noise:
        L = np.array([nr_delay_sim(
                comp=f, 
                time_window_ref=time_window_ref, 
                time_window_comp=time_window_comp, 
                one_cycle=one_cycle,
                ramp=ramp,
                noise=False
            )
            [0] for f in F])
        LS = L*0
    else:
        Ls = []
        for i in range(250):
            L = np.array([nr_delay_sim(
                    comp=f, 
                    time_window_ref=time_window_ref, 
                    time_window_comp=time_window_comp, 
                    one_cycle=one_cycle,
                    ramp=ramp,
                    noise=True
                )
                [0] for f in F])
            Ls.append(L)
        L = np.stack(Ls)
        
        L = L.mean(0)
        LS = L.std(0)
    
    return F, L, LS #np.concatenate([F2, F]), np.concatenate([L2, L])


def nr_delay_sim(
    reference=10, 
    comp=50, 
    time_window_ref=0.2, 
    time_window_comp=0.2,
    one_cycle=True,
    ramp=False,
    noise=False,
    ):
    fs = 1000
    dt = 1/fs
    dur = 2       
    t = np.linspace(-dur/2, dur/2, 2*fs)
    data_ref = np.sin(2*np.pi*reference*t)
    data_ref[t<0] = 0
    data_comp = np.sin(2*np.pi*comp*t)
    data_comp[t<0] = 0
    cycles = [reference/(1/time_window_ref), comp/(1/time_window_comp)]
    if one_cycle:
        data_ref[t>((1/reference)*1)] = 0
        data_comp[t>((1/comp)*1)] = 0

    if ramp:
        data_ref[t>=0] = data_ref[t>=0]*t[t>=0]*2
        data_comp[t>=0] = data_comp[t>=0]*t[t>=0]*2
    if noise:   
        data_ref = data_ref+np.random.randn(len(data_ref))*0.1
        data_comp = data_comp+np.random.randn(len(data_comp))*0.1
    

    from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
    power = tfr_array_multitaper(
        np.stack([data_ref, data_comp])[:, None, :], 
        fs, 
        [reference, comp], 
        cycles, 
        time_bandwidth=2,
        output='power'
    )
    power_ref = power[0, 0, 0, :]
    power_ref = power_ref/power_ref.max()
    power_comp = power[1, 0, 1, :]
    power_comp = power_comp/power_comp.max()

    #compute cross correlation:
    t_ref = ((-0.5)<=t) & (t<=(0.5))
    #lags = np.linspace(-0.2, 0.2, 501)
    lag_idx = [np.roll(t_ref, l) for l in np.arange(-150, 151, 2)]
    lags = [-t[l].mean() for l in lag_idx]
    
    cc = [np.corrcoef(power_ref[t_ref], power_comp[l])[0,1] for l in lag_idx]
    shift = lags[np.argmax(cc)]

    return shift, t, data_ref, data_comp, power_ref, power_comp, lags, cc


def nr_figureS6(df=None, stats=None):
    nr_figureS6A(df=df, stats=stats)
    nr_figureS6B()


def nr_figureS6A(df=None, stats=None):
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_stats_20190516.pickle"
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190308.hdf"
        )

    with mpl.rc_context(rc=rc):
        figure(figsize=(7.5, 7.5 / 2))
        from conf_analysis.meg import srtfr

        # gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        fig = srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            # gs=gs[0, 0],
            stats=stats,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        # [left, bottom, width, height]
        cax = fig.add_axes([0.74, 0.2, 0.1, 0.015])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            orientation="horizontal",
            ticks=[-25, 0, 25],
            drawedges=False,
            label="% Power change",
        )
        cb.outline.set_visible(False)
        sns.despine(ax=cax)

    # fig.colorbar(im, cax=cax)
    #savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS6A.pdf",
    #    dpi=1200,
    #    bbox_inches="tight",
    #)
    return fig, df, stats


def nr_figureS6B(dconf=None):
    """
    Labels:
    conf = True -> Low confidence
    conf = False -> High confidence
    mc = True -> Reference correct
    mc = False -> Test stimulus correct
    """
    figure(figsize=(5, 2.5))
    labels = {
        "['mc', 'conf'](False, False)": "Choice: test stronger; Confidence: high",
        "['mc', 'conf'](False, True)": "Choice: test stronger; Confidence: low",
        "['mc', 'conf'](True, False)": "Choice: test weaker; Confidence: high",        
        "['mc', 'conf'](True, True)": "Choice: test weaker; Confidence: low",
    }
    colors = {
        "['mc', 'conf'](False, False)": "xkcd:orange",
        "['mc', 'conf'](False, True)": "gray",
        "['mc', 'conf'](True, False)": "xkcd:orange",        
        "['mc', 'conf'](True, True)":  "gray",
    }
    ls = {
        "['mc', 'conf'](False, False)": "-",
        "['mc', 'conf'](False, True)": "-",
        "['mc', 'conf'](True, False)": ":",        
        "['mc', 'conf'](True, True)":  ":",
    }

    if dconf is None:
        dconf = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/decoding_choice_split_conf_split_mc.hdf')

    xs = {
        l:pd.pivot_table(index='subject', columns='latency', values='test_roc_auc', data=d.query('epoch=="stimulus" & hemi=="Pair" & cluster=="JWG_M1_LH JWG_M1_RH"')) 
            for l, d in dconf.groupby('split_label')
    }
    x_ref_high =  "['mc', 'conf'](True, False)"
    x_ref_low =  "['mc', 'conf'](True, True)"

    x_stim_high = "['mc', 'conf'](False, False)"
    x_stim_low = "['mc', 'conf'](False, True)"
    

    plotwerr(-(xs[x_ref_high]-0.5)+0.5, label=labels[x_ref_high], color=colors[x_ref_high], linestyle=ls[x_ref_high])
    table_to_data_source_file('S6', 'B', 'choice_weaker_confidence_high', xs[x_ref_high])
    plotwerr(-(xs[x_ref_low]-0.5)+0.5, label=labels[x_ref_low], color=colors[x_ref_low], linestyle=ls[x_ref_low])
    table_to_data_source_file('S6', 'B', 'choice_weaker_confidence_low', xs[x_ref_low])
    draw_sig(plt.gca(), xs[x_ref_high]-xs[x_ref_low], y=0.3, cluster_test=True, linestyle='-')
    draw_sig(plt.gca(), xs[x_ref_high]-xs[x_ref_low], y=0.3, cluster_test=True, color='gray', linestyle=ls[x_ref_low])

    plotwerr((xs[x_stim_high]), label=labels[x_stim_high], color=colors[x_stim_high], linestyle=ls[x_stim_high])
    table_to_data_source_file('S6', 'B', 'choice_stronger_confidence_high', xs[x_stim_high])
    plotwerr((xs[x_stim_low]), label=labels[x_stim_low], color=colors[x_stim_low], linestyle=ls[x_stim_low])
    table_to_data_source_file('S6', 'B', 'choice_stronger_confidence_low', xs[x_stim_low])
    draw_sig(plt.gca(), xs[x_stim_high]-xs[x_stim_low], y=0.35, cluster_test=True, linestyle='-')
    draw_sig(plt.gca(), xs[x_stim_high]-xs[x_stim_low], y=0.35, cluster_test=True, color='gray', linestyle=ls[x_stim_low])

    plt.legend(frameon=False, fontsize=7)
    plt.ylabel('AUC', fontsize=7)
    plt.axhline(0.5, color='k', zorder=-1, lw=1)
    plt.ylim([0.25, 0.75])
    plt.ylabel('Time (s)', fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim([-0.25, 1.5])
    sns.despine()
    #savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_figureS6B.pdf",
    #    dpi=1200,
    #    bbox_inches="tight",
    #)


def nr_compare_ssd_avg_pair_lat(ssd = None, acc=False):
    if acc:
        signal="SSD_acc_contrast"
    else:
        signal="SSD" 
    if ssd is None:
        ssd = dp.get_ssd_data(restrict=False)
    ssd_avg = ssd.test_corr.Averaged.query(
        'signal=="%s" & epoch=="stimulus"'%signal
    )
    ssd_pair = ssd.test_corr.Pair.query(
        'signal=="%s" & epoch=="stimulus"'%signal
    )
    ssd_lat = ssd.test_corr.Lateralized.query(
        'signal=="%s" & epoch=="stimulus"'%signal
    )    
    o_lat = np.round(
        ssd_lat.index.get_level_values("latency") + (ssd_lat.index.get_level_values("sample") / 10),
        3,
    )
    o_lat.name = "time"
    o_pair = np.round(
        ssd_pair.index.get_level_values("latency") + (ssd_pair.index.get_level_values("sample") / 10),
        3,
    )
    o_pair.name = "time"
    o_avg = np.round(
        ssd_avg.index.get_level_values("latency") + (ssd_avg.index.get_level_values("sample") / 10),
        3,
    )
    o_avg.name = "time"
    ssd_lat = ssd_lat.set_index(o_lat, append=True)
    ssd_avg = ssd_avg.set_index(o_avg, append=True)
    ssd_pair = ssd_pair.set_index(o_pair, append=True)
    hull_curves_lat = (ssd_lat
            .groupby('subject')
            .apply(
                lambda x: pd.pivot_table(index='sample', columns='time', data=x).max(0)
                )
            )
    hull_curves_pair = (ssd_pair
            .groupby('subject')
            .apply(
                lambda x: pd.pivot_table(index='sample', columns='time', data=x).max(0)
                )
            )
    hull_curves_avg = (ssd_avg
            .groupby('subject')
            .apply(
                lambda x: pd.pivot_table(index='sample', columns='time', data=x).max(0)
                )
            )

    
    hull_diff = [dp.sps_2lineartime(
            dp.get_ssd_per_sample(ssd.test_corr.Averaged.query('subject==%i'%subject), 'SSD', 'vfcPrimary') - 
            dp.get_ssd_per_sample(ssd.test_corr.Pair.query('subject==%i'%subject), 'SSD', 'vfcPrimary')
            ) for subject in range(1, 16)]
    hds = []
    for subject in range(1, 16):
        o = hull_diff[subject-1].stack().reset_index()
        o.columns = ['latency', 'sample', 'diff']
        o.loc[:, 'subject'] = subject
        hds.append(o)
    hd = pd.concat(hds)
    figure(figsize=(7.5, 4))
    sd = []
    for s, ds in hd.groupby('sample'):
        plt.subplot(2, 5, s+1)
        X = pd.pivot_table(data=ds, index='subject', columns='latency', values='diff')
        #print(X.values.ravel().mean(), X.values.ravel().std())
        xm = X.loc[:, 0:1.2].mean(1).to_frame()
        xm.loc[:,'sample'] = s
        sd.append(xm)

        plotwerr(X)
        draw_sig(plt.gca(), X)
        draw_sig(plt.gca(), X, y=-0.01, cluster_test=True)
        plt.ylim([-0.03, 0.03])
        plt.xlim([-0.2, 1.3])
        plt.axhline(0,color='gray', lw=1, zorder=-1)
        plt.xlabel('Time (s)', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylabel('Average - Pair\ncorrelation', fontsize=7)

    plt.tight_layout()
    sns.despine()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/revision/figures/nr_decoding_differences_average_pair_V1.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return hull_curves_avg, hull_curves_lat, hull_curves_pair, pd.concat(sd)


