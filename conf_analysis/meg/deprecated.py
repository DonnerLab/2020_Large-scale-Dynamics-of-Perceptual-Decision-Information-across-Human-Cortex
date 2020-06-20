def plot_signal_comp(df, latency=1.2,  xlim=[0.49, 0.85], auc_cutoff=0.6, 
     gs=None, idx=None):
    """
    Call this function to plot all decoding values at latency=x in df
    during the response period.

    The plot will compare Averaged/Lateralized and MIDC_split with
    CONF_signed and CONF_unsigned.

    df should be output from get_decoding_data().test_roc_auc

    Statistics are done using Bayesian estimation and comparison
    against pre-stimulus baseline data (latency=0).

    For now only setting latency=0 works reliably with the stats
    (because latency = 0 -> pre stimulus baseline and interesting
    time point in response period.)
    """

    #df = df.query("latency==%f" % latency)
    df_interest = df.query("epoch=='stimulus'").query("latency==%f" % latency)
    df_base = df.query("epoch=='stimulus'").query("-0.25<latency<0")
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1,2)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1,2, gs)
    #plt.figure(figsize=(5.5, 10))
    plt.subplot(gs[0, 1])
    
    idx, pval_agreement = plot_max_per_area(
        df_interest.Lateralized,
        df_base.Lateralized,
        "MIDC_split",
        ["CONF_signed", "CONF_unsigned"],
        text="right",
        sorting=idx   
    )

    # plt.cla()
    plt.axvline(0.5, color="k", lw=0.5)
    plt.xlim(xlim)
    plt.title("Lateralized H")
    plt.subplot(gs[0,0])
    agr, pval_agreement2 = plot_max_per_area(
        df_interest.Averaged,
        df_base.Averaged,
        "MIDC_split",
        ["CONF_signed", "CONF_unsigned"],
        sorting=idx,
        text="none",
    )
    plt.title("Averaged H")
    plt.xlim(xlim[::-1])
    plt.axvline(0.5, color="k", lw=0.5)
    return idx, {"lateralized": pval_agreement, "averaged": pval_agreement2}


def get_uncertainty(df, df_ref, signal, name, latency):
    """
    Compute HDI for comparing data in df against df_ref. 
    Both dfs should contain one time point, df the one of interest,
    df_ref a baseline time-point (e.g. pre-stimulus). 
    """
    df = df.query("signal=='%s' " % (signal)).loc[:, name]
    #df = df.groupby(["subject"]).mean()
    df = pd.pivot_table(data=df.reset_index(), index="subject", columns="latency",
                    values=name)
    df_ref = df_ref.query("signal=='%s'" % (signal)).loc[:, name]
    df_ref = pd.pivot_table(data=df_ref.reset_index(), index="subject", columns="latency",
                    values=name)
    #df_ref = df_ref.groupby(["subject"]).mean()
    mu, mu_hdi, phdi, _, _ = get_baseline_posterior(df, df_ref)
    #m  = df.mean().values
    #s= df.std().values 
    return mu, mu_hdi, phdi


def plot_scatter_with_sorting(
    df, df_ref, signal, values, latency, names, uncertainty=True, cmap="Greens"
):
    """
    For statistics to work correctly df and df_ref should have one time-point only.

    Plot decoding values in df.
    """
    cm = matplotlib.cm.get_cmap(cmap)
    l = 1 + 0 * np.abs(latency.copy())  # Should be min 0 now
    pval_agreement = 0
    cnt = 0
    colors = cm(l[::-1] / 2)
    from scipy.stats import ttest_1samp

    for x, (name, latency) in enumerate(zip(names, latency)):        
        mu, mu_ref, hpd = get_uncertainty(df, df_ref, signal, name, latency)
        hpd = 0.5 + hpd.ravel()
        df2 = df.query("signal=='%s' & latency==%f" % (signal, latency)).loc[:, name]
        df2 = df2.groupby(["subject"]).mean()
        tval, pval = ttest_1samp(df2.values.ravel(), 0.5)
        
        plt.plot(hpd, [x, x], color=colors[x], lw=0.5)
        if (hpd[0] <= 0.5) and (0.5 <= hpd[1]):
            if pval > 0.05:
                pval_agreement += 1
            plt.plot(values[x], x, "o", fillstyle="none", color=colors[x])
        else:
            if pval <= 0.05:
                pval_agreement += 1
            plt.plot(values[x], x, "o", fillstyle="full", color=colors[x])
        cnt += 1
    return pval_agreement / cnt


def plot_max_per_area(
    df,
    df_ref,
    signal,
    auxil_signal=None,
    auxmaps=["Reds", "Blues"],
    sorting=None,
    text="right",
):
    """
    Plot max for each area and color code latency.
    """
    df_main = df.groupby(["signal", "latency"]).mean().query('signal=="%s"' % signal)
    idxmax = df_main.idxmax()
    xticks = []
    values = df_main.max().values
    pval_agreement = {}
    if sorting is not None:
        idx = sorting
    else:
        idx = np.argsort(values)

        # Filter to max_val >= 0.55
        i = 0
        for i in idx:
            if values[i] > 0.55:
                break
        idx = idx[i:]
    latency = np.array([x[1] for _, x in idxmax.iteritems()])
    agr = plot_scatter_with_sorting(
        df, df_ref, signal, values[idx], latency[idx], idxmax.index.values[idx]
    )
    pval_agreement[signal] = agr
    xticks = np.array(
        [
            name.replace("NSWFRONT_", "").replace("HCPMMP1_", "")
            for name in idxmax.index.values
        ]
    )
    if text == "left":
        dx = -0.005
        text = "right"
    elif text == "right":
        dx = +0.025
        text = "left"
    elif text == "center":
        dx = 0
    if not text == "none":
        for x, y, t in zip(values[idx], np.arange(len(values)), xticks[idx]):
            if text == "center":
                x = 0.45
            plt.text(
                x + dx,
                y,
                t,
                verticalalignment="center",
                horizontalalignment=text,
                fontsize=8,
            )

    if auxil_signal is not None:
        for signal, cmap in zip(auxil_signal, auxmaps):
            df_aux = (
                df.groupby(["signal", "latency"]).mean().query('signal=="%s"' % signal)
            )
            idxmax_aux = df_aux.idxmax()
            latency_aux = np.array([x[1] for _, x in idxmax_aux.iteritems()])
            values_aux = df_aux.max().values
            agr = plot_scatter_with_sorting(
                df,
                df_ref,
                signal,
                values_aux[idx],
                latency_aux[idx],
                idxmax_aux.index.values[idx],
                cmap=cmap,
            )
            pval_agreement[signal] = agr

    import seaborn as sns
    sns.despine(left=True, ax=plt.gca())
    plt.yticks([], [])
    plt.xlabel("AUC")
    return idx, pval_agreement
