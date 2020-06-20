
def make_brain_plots(data, ssd_view=['cau']):
    # 1 AUC
    auc_limits = {'MIDC_split': (0.3, 0.7), 'MIDC_nosplit': (0.3, 0.7),
                  'CONF_signed': (0.3, 0.7), 'CONF_unsigned': (.3, 0.7),
                  'CONF_unsign_split': (.3, 0.7),
                  'SIDE_nosplit': (0.3, 0.7)}

    df = data.test_roc_auc.Pair.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.15, 0.2),
                         limits=auc_limits, epoch='stimulus', measure='auc')
    df = data.test_roc_auc.Pair.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=auc_limits, epoch='response', measure='auc')

    df = data.test_roc_auc.loc[:, atype].query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.425, 0.475),
                         limits=auc_limits, epoch='response_late_' + 'atype', measure='auc')

    acc_limits = {'MIDC_split': (-0.2, 0.2), 'MIDC_nosplit': (-0.2, 0.2),
                  'CONF_signed': (-0.2, 0.2), 'CONF_unsigned': (-.2, 0.2),
                  'SIDE_nosplit': (-0.2, 0.2)}
    df = data.test_accuracy.Pair.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='stimulus', measure='accuracy')
    df = data.test_accuracy.Pair.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='response', measure='accuracy')

    data_ssd = filter_latency(data.test_slope.Pair.query(
        'epoch=="stimulus" & (signal=="SSD") & Classifier=="Ridge"'), 0.18, 0.19)
    for sample, sd in data_ssd.groupby('sample'):
        ssd_limits = {'SSD': (-0.08, 0.08)}
        plot_summary_results(sd, limits=ssd_limits,
                             epoch='stimulus',
                             measure='slope' + '_sample%i' % sample,
                             views=ssd_view)


def plot_summary_results(data, cmap='RdBu_r',
                         limits={'MIDC_split': (0.3, 0.7),
                                 'MIDC_nosplit': (0.3, 0.7),
                                 'CONF_signed': (0.3, 0.7),
                                 'CONF_unsigned': (.3, 0.7),
                                 'SIDE_nosplit': (0.3, 0.7),
                                 'CONF_unsign_split': (.3, 0.7),
                                 'SSD': (-0.05, 0.05),
                                 'SSD_acc_contrast': (-0.05, 0.05),
                                 'SSD_acc_contrast_diff': (-0.05, 0.05),
                                 'SSD_delta_contrast': (-0.05, 0.05)},
                         ex_sub='S04', measure='auc', epoch='response',
                         classifier='svc',
                         views=[['par', 'fro'], ['lat', 'med']]):
    from pymeg import roi_clusters as rois, source_reconstruction as sr

    # labels = sr.get_labels(ex_sub)
    labels = sr.get_labels(ex_sub)
    lc = rois.labels_to_clusters(labels, rois.all_clusters, hemi='lh')

    for signal, dsignal in data.groupby('signal'):
        vmin, vmax = limits[signal]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colortable = cm.get_cmap(cmap)
        cfunc = lambda x: colortable(norm(x))
        brain = plot_one_brain(dsignal, signal, lc, cfunc, ex_sub=ex_sub,
                               measure=measure, classifier=classifier,
                               epoch=epoch, views=views)
    # return brain


#@memory.cache
def plot_one_brain(dsignal, signal, lc, cmap, ex_sub='S04', classifier='SCVlin',
                   epoch='response', measure='auc', views=[['par', 'fro'], ['lat', 'med']]):
    from surfer import Brain
    print('Creating Brain')
    brain = Brain(ex_sub, 'lh', 'inflated',  views=['lat'], background='w')
    # subjects_dir='/Users/nwilming/u/freesurfer_subjects/')
    print('Created Brain')
    ms = dsignal.mean()
    if (signal == 'CONF_signed') and (measure == 'accuracy'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    if (signal == 'SSD'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    else:
        plot_labels_on_brain(brain, lc, ms, cmap)
    brain.save_montage('/Users/nwilming/Desktop/%s_montage_%s_%s_%s.png' %
                       (signal, measure, classifier, epoch), views)
    return brain


def plot_labels_on_brain(brain, labels, data, cmap):
    already_plotted = []
    for label in data.index.values:
        for clustername, lobjects in labels.items():
            if clustername == label:
                for l0 in lobjects:
                    if any(l0.name == x for x in already_plotted):
                        import pdb
                        pdb.set_trace()
                    # print(('Addding', l0.name, cmap(value)))
                    already_plotted.append(l0.name)
                    value = data.loc[label]
                    l0.color = cmap(value)
                    brain.add_label(l0, color=cmap(value), alpha=0.8)

    brain.save_image('test.png')


def plot_brain_color_legend(palette):
    from surfer import Brain
    from pymeg import atlas_glasser as ag
    from pymeg import source_reconstruction as sr

    labels = sr.get_labels(subject='S04', filters=[
        '*wang*.label', '*JWDG*.label'], annotations=['HCPMMP1'])
    labels = sr.labels_exclude(labels=labels, exclude_filters=[
        'wang2015atlas.IPS4', 'wang2015atlas.IPS5', 'wang2015atlas.SPL',
        'JWDG_lat_Unknown'])
    labels = sr.labels_remove_overlap(
        labels=labels, priority_filters=['wang', 'JWDG'])
    lc = ag.labels2clusters(labels)
    brain = Brain('S04', 'lh', 'inflated',  views=['lat'], background='w')
    for cluster, labelobjects in lc.items():
        if cluster in palette.keys():
            color = palette[cluster]
            for l0 in labelobjects:
                if l0.hemi == 'lh':
                    brain.add_label(l0, color=color, alpha=1)
    brain.save_montage('/Users/nwilming/Dropbox/UKE/confidence_study/brain_colorbar.png',
                       [['par', 'fro'], ['lat', 'med']])
    return brain


def plot_signals(data, measure, classifier='svm', ylim=(0.45, 0.75)):
    for epoch, de in data.groupby('epoch'):
        g = plot_by_signal(de)
        g.set_ylabels(r'$%s$' % measure)
        g.set_xlabels(r'$time$')
        g.set(ylim=ylim)
        plt.savefig('/Users/nwilming/Desktop/%s_%s_%s_decoding.pdf' %
                    (measure, classifier, epoch))


def plot_decoding_results(data, signal, area,
                          stim_ax=None, resp_ax=None,  color='b',
                          offset=0):
    '''
    Data is a df that has areas as columns and at least subjct, classifier, latency and signal as index.
    Values of the dataframe encode the measure of choice to plot.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    if stim_ax is None:
        stim_ax = plt.gca()
    if resp_ax is None:
        stim_ax = plt.gca()
    data = data.loc[:, area]
    select_string = 'signal=="%s"' % (signal)
    areaname = (str(area).replace('vfc', '')
                .replace('-lh', '')
                .replace('-rh', '')
                .replace('_Havg', '')
                .replace('_Lateralized', ''))
    data = data.reset_index().query(select_string)
    if '_split' in signal:
        data = data.groupby(['subject', 'epoch', 'latency']
                            ).mean().reset_index()
    stimulus = data.query('epoch=="stimulus"').reset_index()
    stimulus.loc[:, area] += offset

    response = data.query('epoch=="response"').reset_index()
    response.loc[:, area] += offset
    stim_ax = plt.subplot(stim_ax)
    sns.tsplot(stimulus, time='latency', value=area,
               unit='subject', ax=stim_ax, color=color)
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')
    dx, dy = np.array([0.0, 0.0]), np.array([.5, 0.75])
    plt.plot(dx, dy + offset, color='k')
    plt.text(-0.75, 0.6 + offset, areaname)
    plt.yticks([])
    plt.ylabel('')
    resp_ax = plt.subplot(resp_ax)

    sns.tsplot(response, time='latency', value=area,
               unit='subject', ax=resp_ax, color=color)
    plt.plot(dx, dy + offset, color='k')
    plt.yticks([])
    plt.ylabel('')
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')
