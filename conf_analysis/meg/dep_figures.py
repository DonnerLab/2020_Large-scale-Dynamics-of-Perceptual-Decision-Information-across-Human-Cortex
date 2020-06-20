
def _figure2(df=None):
    """
    Plot Decoding results
    """
    if df is None:
        df = dp.get_decoding_data()
    palette = dp.get_area_palette()
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        plt.figure(figsize=(8, 5.5))
        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "Reds", "CONF_unsigned": "Blues", "CONF_signed": "Greens"},
            {
                "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": df.test_roc_auc.Lateralized,
            },
        )
        plotter.plot()

        plt.tight_layout()
    try:
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig2.pdf"
        )
    except FileNotFoundError:
        savefig("/home/student/n/nwilming/conf_fig2.pdf")
    return df


def _figure3(df=None, quick=False):
    """
    Reproduce state space plot and decoding of all areas at
    t = peak sensory decoding
    """
    import pymc3 as pm
    import seaborn as sns

    colors = {"MIDC_split": "r", "CONF_unsigned": "b", "CONF_signed": "g"}
    figure(figsize=(8, 4.5))
    if df is None:
        df = dp.get_decoding_data()
    gs = matplotlib.gridspec.GridSpec(2, 5, hspace=0.5, wspace=0.3)
    subplot(gs[:, :2])

    """
    State space plot:
    """
    if not quick:
        dp.state_space_plot(
            df.test_roc_auc.Lateralized,
            "MIDC_split",
            "CONF_unsigned",
            df_b=df.test_roc_auc.Lateralized,
            a_max=0.525,
            color=sns.xkcd_rgb["purple"],
        )
        dp.state_space_plot(
            df.test_roc_auc.Lateralized,
            "MIDC_split",
            "CONF_signed",
            df_b=df.test_roc_auc.Lateralized,
            a_max=0.525,
            color=sns.xkcd_rgb["orange"],
        )
        dp.state_space_plot(
            df.test_roc_auc.Averaged,
            "MIDC_split",
            "CONF_signed",
            df_b=df.test_roc_auc.Averaged,
            a_max=0.525,
            color="k",
        )
    plt.ylim([0.45, 0.75])
    plt.xlim([0.45, 0.75])
    plt.xlabel("AUC")
    plt.ylabel("AUC")
    sns.despine(ax=plt.gca())

    """
    Plot comparison of values for Stim locked episodes.

    """
    plt.plot([0.5, 0.75], [0.5, 0.75], "k", lw=0.5)
    sorting, ax_diff_stim = dp.plot_signal_comp(
        df,
        latency=1.2,
        auc_cutoff=0.515,
        gs=gs[:, 2:],
        xlim=[0.45, 0.9],
        xlim_avg=[0.49, 0.65],
        colors=colors,
    )

    # Get diff between absolute conf between averaged H and lateralized H
    signals, subjects, areas, dataA = dp.get_signal_comp_data(
        df.test_roc_auc.Averaged, 1.2, "stimulus"
    )
    signals, subjects, areas, dataL = dp.get_signal_comp_data(
        df.test_roc_auc.Lateralized, 1.2, "stimulus"
    )

    from conf_analysis.meg import stats

    k, mdl = stats.auc_get_sig_cluster_group_diff_posterior(dataA, dataL)
    hpd = pm.stats.hpd(k.get_values("mu_diff"))
    signal_delta = {"MIDC_split": 0.15, "CONF_unsigned": 0, "CONF_signed": -0.15}
    for j, signal in enumerate(signals):
        mean = (dataL[j, :, :] - dataA[j, :, :]).mean(-1).squeeze()
        for i, s in enumerate(sorting):
            ax_diff_stim.plot(
                mean[s], i + signal_delta[signal], ".", color=colors[signal]
            )
            ax_diff_stim.plot(
                hpd[s, j, :],
                np.array([i, i]) + signal_delta[signal],
                color=colors[signal],
            )

    sns.despine(left=True, ax=ax_diff_stim)
    ax_diff_stim.set_xlim([-0.2, 0.2])
    ax_diff_stim.set_xticks([-0.1, 0, 0.1], minor=False)
    ax_diff_stim.set_xticklabels(["-0.1", "", "0.1"])
    ax_diff_stim.axvline(0, color="k", lw=0.5)
    ax_diff_stim.set_yticks([])
    plt.xlabel(r"$\Delta$AUC")
    plt.title("Diff.")

    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig3.pdf",
        bbox_inches="tight",
    )
    return k, mdl


def supp_figure_Z(df=None):
    colors = {"MIDC_split": "r", "CONF_unsigned": "b", "CONF_signed": "g"}
    import pymc3 as pm

    figure(figsize=(7.5, 5))
    if df is None:
        df = dp.get_decoding_data()
    gs = matplotlib.gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.3)

    ax = subplot(gs[0, :])

    high = 0.7
    low = 0.5
    norm = matplotlib.colors.Normalize(vmin=low, vmax=high)

    colors = {
        "MIDC_split": lambda x: matplotlib.cm.get_cmap("Reds")(norm(x)),
        "CONF_unsigned": lambda x: matplotlib.cm.get_cmap("Blues")(norm(x)),
        "CONF_signed": lambda x: matplotlib.cm.get_cmap("Greens")(norm(x)),
    }
    idx = dp._plot_signal_comp(
        df.test_roc_auc.Pair,
        1.217,
        None,
        colors,
        "Pair",
        auc_cutoff=0.5,
        ax=ax,
        horizontal=True,
        plot_labels=False,
        color_by_cmap=True,
    )
    ax.set_xlabel("ROI")
    ax.set_ylabel("AUC")
    ax.set_title("")
    ax.set_ylim([0.49, 0.7])
    y = np.linspace(0.49, 0.7, 250)
    x = y * 0 - 3
    print(xlim())
    xlim([-4, 154])
    scatter(x, y, s=0.5, c=colors["MIDC_split"](y))
    scatter(x + 3, y, s=0.5, c=colors["CONF_unsigned"](y))
    scatter(x + 1.5, y, s=0.5, c=colors["CONF_signed"](y))
    sns.despine(bottom=True, ax=ax)

    subplot(gs[1, 0])
    df_t = df.query("latency==1.217").test_roc_auc.Pair.groupby("signal").mean()

    img = _get_lbl_annot_img(
        dict(df_t.loc["MIDC_split"]),
        low=low,
        high=high,
        views=[["lat"], ["med"]],
        colormap="Reds",
    )
    imshow(img)
    xticks([])
    yticks([])
    sns.despine(left=True, bottom=True, ax=gca())

    subplot(gs[1, 1])
    img = _get_lbl_annot_img(
        dict(df_t.loc["CONF_signed"]),
        low=low,
        high=high,
        views=[["lat"], ["med"]],
        colormap="Greens",
    )
    imshow(img)
    xticks([])
    yticks([])
    sns.despine(left=True, bottom=True, ax=gca())

    subplot(gs[1, 2])
    img = _get_lbl_annot_img(
        dict(df_t.loc["CONF_unsigned"]),
        low=low,
        high=high,
        views=[["lat"], ["med"]],
        colormap="Blues",
    )
    imshow(img)
    xticks([])
    yticks([])
    sns.despine(left=True, bottom=True, ax=gca())

    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/sup_fig_Z.pdf",
        bbox_inches="tight",
        dpi=1200,
    )


def sup_figure1(df=None):
    import seaborn as sns

    if df is None:
        df = dp.get_decoding_data(restrict=False, ogl=True)
    stim = df.test_roc_auc.Pair.query('epoch=="stimulus"')
    resp = df.test_roc_auc.Pair.query('epoch=="response"')

    import matplotlib

    norm = matplotlib.colors.Normalize(vmin=0.3, vmax=0.7)
    cm = matplotlib.cm.get_cmap("RdBu_r")
    cnt = 0
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        plt.figure(figsize=(9, 3))
        gs = matplotlib.gridspec.GridSpec(3, 13)

        for j, signal in enumerate(["MIDC_split", "CONF_signed", "CONF_unsigned"]):
            # First plot Stimulus locked
            latencies = stim.index.get_level_values("latency").unique()
            stim_latencies = np.arange(0.3, 1.3, 0.1)
            for i, latency in enumerate(stim_latencies):
                target_lat = latencies[np.argmin(np.abs(latencies - latency))]
                d = (
                    stim.query('latency==%f & signal=="%s"' % (target_lat, signal))
                    .groupby("signal")
                    .mean()
                )
                palette = {k: d[k].values[0] for k in d.columns}
                img = _get_img(palette, low=0.3, high=0.7)
                plt.subplot(gs[j, i])
                plt.imshow(img, aspect="equal")
                plt.xticks([])
                plt.yticks([])
                sns.despine(left=True, bottom=True)
                if signal == "MIDC_split":
                    n = ("%0.1f" % target_lat).replace("0", "")
                    title(r"t=%s" % n)
                if i == 0:
                    ylabel(signal)
            # Now response locked
            latencies = resp.index.get_level_values("latency").unique()
            for i, latency in enumerate(np.arange(-0.2, 0.01, 0.1)):
                target_lat = latencies[np.argmin(np.abs(latencies - latency))]
                d = (
                    resp.query('latency==%f & signal=="%s"' % (target_lat, signal))
                    .groupby("signal")
                    .mean()
                )
                palette = {k: d[k].values[0] for k in d.columns}
                img = _get_img(palette, low=0.3, high=0.7)
                plt.subplot(gs[j, i + len(stim_latencies)])
                plt.imshow(img, aspect="equal")
                plt.xticks([])
                plt.yticks([])
                sns.despine(left=True, bottom=True)
                if signal == "MIDC_split":
                    n = ("%0.1f" % target_lat).replace("0", "")
                    if n == ".":
                        n = "0"
                    title(r"t=%s" % n)
    plt.savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_figure1.pdf",
        dpi=1200,
    )
    return df
