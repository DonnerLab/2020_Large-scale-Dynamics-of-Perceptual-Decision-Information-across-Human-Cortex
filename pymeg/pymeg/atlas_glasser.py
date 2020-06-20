import mne
import os


"""
Get HCP MMP Parcellation for a set of subjects
"""


def get_hcp(subjects_dir):
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)


def get_hcp_annotation(subjects_dir, subject):
    for hemi in ["lh", "rh"]:
        # transform atlas to individual space:
        cmd = "mris_apply_reg --src-annot {} --trg {} --streg {} {}".format(
            os.path.join(
                subjects_dir, "fsaverage", "label", "{}.HCPMMP1.annot".format(hemi)
            ),
            os.path.join(
                subjects_dir, subject, "label", "{}.HCPMMP1.annot".format(hemi)
            ),
            os.path.join(
                subjects_dir, "fsaverage", "surf", "{}.sphere.reg".format(hemi)
            ),
            os.path.join(subjects_dir, subject, "surf", "{}.sphere.reg".format(hemi)),
        )
        os.system(cmd)


def get_hcp_labels(subjects_dir, subjects):
    """
    Downloads HCP MMP Parcellation and applies it to a set of subjects

    Arguments
    =========
    subjects_dir: str
        Path of freesurfer subjects dir
    subjects: list
        List of subject IDs (need to correspond to folders in
        freesurfer subject dir.)
    """

    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

    for subj in subjects:
        for hemi in ["lh", "rh"]:

            # transform atlas to individual space:
            cmd = "mris_apply_reg --src-annot {} --trg {} --streg {} {}".format(
                os.path.join(
                    subjects_dir,
                    "fsaverage",
                    "label",
                    "{}.HCPMMP1_combined.annot".format(hemi),
                ),
                os.path.join(
                    subjects_dir,
                    subj,
                    "label",
                    "{}.HCPMMP1_combined.annot".format(hemi),
                ),
                os.path.join(
                    subjects_dir, "fsaverage", "surf", "{}.sphere.reg".format(hemi)
                ),
                os.path.join(subjects_dir, subj, "surf", "{}.sphere.reg".format(hemi)),
            )
            os.system(cmd)

            # unpack into labels:
            cmd = "mri_annotation2label --subject {} --hemi {} --labelbase {} --annotation {}".format(
                subj,
                hemi,
                "{}.HCPMMP1_combined".format(hemi),
                "HCPMMP1_combined".format(hemi),
            )
            os.system(cmd)

            # rename in alphabetical order...
            orig_names = [
                "???",
                "Anterior Cingulate and Medial Prefrontal Cortex",
                "Auditory Association Cortex",
                "Dorsal Stream Visual Cortex",
                "DorsoLateral Prefrontal Cortex",
                "Early Auditory Cortex",
                "Early Visual Cortex",
                "Inferior Frontal Cortex",
                "Inferior Parietal Cortex",
                "Insular and Frontal Opercular Cortex",
                "Lateral Temporal Cortex",
                "MT+ Complex and Neighboring Visual Areas",
                "Medial Temporal Cortex",
                "Orbital and Polar Frontal Cortex",
                "Paracentral Lobular and Mid Cingulate Cortex",
                "Posterior Cingulate Cortex",
                "Posterior Opercular Cortex",
                "Premotor Cortex",
                "Primary Visual Cortex (V1)",
                "Somatosensory and Motor Cortex",
                "Superior Parietal Cortex",
                "Temporo-Parieto-Occipital Junction",
                "Ventral Stream Visual Cortex",
            ]

            new_names = [
                "23_inside",
                "19_cingulate_anterior_prefrontal_medial",
                "11_auditory_association",
                "03_visual_dors",
                "22_prefrontal_dorsolateral",
                "10_auditory_primary",
                "02_visual_early",
                "21_frontal_inferior",
                "17_parietal_inferior",
                "12_insular_frontal_opercular",
                "14_lateral_temporal",
                "05_visual_lateral",
                "13_temporal_medial",
                "20_frontal_orbital_polar",
                "07_paracentral_lob_mid_cingulate",
                "18_cingulate_posterior",
                "09_opercular_posterior",
                "08_premotor",
                "01_visual_primary",
                "06_somatosensory_motor",
                "16_parietal_superior",
                "15_temporal_parietal_occipital_junction",
                "04_visual_ventral",
            ]

            for o, n, i in zip(orig_names, new_names, ["%.2d" % i for i in range(23)]):
                os.rename(
                    os.path.join(
                        subjects_dir,
                        subj,
                        "label",
                        "{}.HCPMMP1_combined-0{}.label".format(hemi, i),
                    ),
                    os.path.join(
                        subjects_dir,
                        subj,
                        "label",
                        "{}.HCPMMP1_{}.label".format(hemi, o),
                    ),
                )
                os.rename(
                    os.path.join(
                        subjects_dir,
                        subj,
                        "label",
                        "{}.HCPMMP1_{}.label".format(hemi, o),
                    ),
                    os.path.join(
                        subjects_dir,
                        subj,
                        "label",
                        "{}.HCPMMP1_{}.label".format(hemi, n),
                    ),
                )


def get_JWDG_labels(subject, subjects_dir):
    import glob

    lh_labels = glob.glob(os.path.join(subjects_dir, "fsaverage", "label", "lh.JWDG*"))
    rh_labels = glob.glob(os.path.join(subjects_dir, "fsaverage", "label", "rh.JWDG*"))

    mni_reg_file = os.path.join(
        subjects_dir, subject, "mri", "transforms", "reg.mni152.2mm.lta"
    )
    if not os.path.isfile(mni_reg_file):
        print("Doing mni152reg")
        os.system("mni152reg --s %s" % subject)

    for hemi, labels in zip(["lh", "rh"], [lh_labels, rh_labels]):
        for label in labels:
            align_command = "mri_label2label \
                 --srclabel {label} \
                 --srcsubject fsaverage \
                 --trgsubject {subject} \
                 --regmethod surface \
                 --trglabel {base_name} \
                 --hemi {hemi}".format(
                label=label, subject=subject, base_name=label.split("/")[-1], hemi=hemi
            )
            print(align_command)
            os.system(align_command)


def get_clusters():
    # fmt: off
    visual_field_clusters = {
        'vfcPrimary': [
            u'lh.wang2015atlas.V1d-lh', u'rh.wang2015atlas.V1d-rh',
            u'lh.wang2015atlas.V1v-lh', u'rh.wang2015atlas.V1v-rh',
        ],
        'vfcEarly': [
            u'lh.wang2015atlas.V2d-lh', u'rh.wang2015atlas.V2d-rh',
            u'lh.wang2015atlas.V2v-lh', u'rh.wang2015atlas.V2v-rh',
            u'lh.wang2015atlas.V3d-lh', u'rh.wang2015atlas.V3d-rh',
            u'lh.wang2015atlas.V3v-lh', u'rh.wang2015atlas.V3v-rh',
            u'lh.wang2015atlas.hV4-lh', u'rh.wang2015atlas.hV4-rh',
        ],
        'vfcVO': [
            u'lh.wang2015atlas.VO1-lh', u'rh.wang2015atlas.VO1-rh',
            u'lh.wang2015atlas.VO2-lh', u'rh.wang2015atlas.VO2-rh',
        ],
        'vfcPHC': [
            u'lh.wang2015atlas.PHC1-lh', u'rh.wang2015atlas.PHC1-rh',
            u'lh.wang2015atlas.PHC2-lh', u'rh.wang2015atlas.PHC2-rh',
        ],
        'vfcV3ab': [
            u'lh.wang2015atlas.V3A-lh', u'rh.wang2015atlas.V3A-rh',
            u'lh.wang2015atlas.V3B-lh', u'rh.wang2015atlas.V3B-rh',
        ],
        'vfcTO': [
            u'lh.wang2015atlas.TO1-lh', u'rh.wang2015atlas.TO1-rh',
            u'lh.wang2015atlas.TO2-lh', u'rh.wang2015atlas.TO2-rh',
        ],
        'vfcLO': [
            u'lh.wang2015atlas.LO1-lh', u'rh.wang2015atlas.LO1-rh',
            u'lh.wang2015atlas.LO2-lh', u'rh.wang2015atlas.LO2-rh',
        ],
        'vfcIPS01': [
            u'lh.wang2015atlas.IPS0-lh', u'rh.wang2015atlas.IPS0-rh',
            u'lh.wang2015atlas.IPS1-lh', u'rh.wang2015atlas.IPS1-rh',
        ],
        'vfcIPS23': [
            u'lh.wang2015atlas.IPS2-lh', u'rh.wang2015atlas.IPS2-rh',
            u'lh.wang2015atlas.IPS3-lh', u'rh.wang2015atlas.IPS3-rh',
            # u'lh.wang2015atlas.IPS4-lh', u'rh.wang2015atlas.IPS4-rh',
            # u'lh.wang2015atlas.IPS5-lh', u'rh.wang2015atlas.IPS5-rh',
        ],
        'vfcFEF': [
            u'lh.wang2015atlas.FEF-lh', u'rh.wang2015atlas.FEF-rh',
        ],
    }

    jwg_clusters = {
        'JWG_aIPS': ['lh.JWDG.lr_aIPS1-lh', 'rh.JWDG.lr_aIPS1-rh',
                     'lh.JWG_lat_aIPS-lh', 'rh.JWG_lat_aIPS-rh',
                     'lh.JWG_aIPS1-lh', 'rh.JWG_aIPS1-rh', ],
        'JWG_IPS_PCeS': ['lh.JWDG.lr_IPS_PCes-lh', 'rh.JWDG.lr_IPS_PCes-rh',
                         'lh.JWG_lat_IPS_PCeS-lh', 'rh.JWG_lat_IPS_PCeS-rh',
                         'lh.JWG_IPS_PCeS-lh', 'rh.JWG_IPS_PCeS-rh', ],
        'JWG_M1': ['lh.JWDG.lr_M1-lh', 'rh.JWDG.lr_M1-rh',
                   'lh.JWG_lat_M1-lh', 'rh.JWG_lat_M1-rh',
                   'lh.JWG_M1-lh', 'rh.JWG_M1-rh', ],
    }

    glasser_clusters = {
        'HCPMMP1_visual_primary': (
            ['L_{}_ROI-lh'.format(area) for area in ['V1']] +
            ['R_{}_ROI-rh'.format(area) for area in ['V1']]
        ),
        'HCPMMP1_visual_dors': (
            ['L_{}_ROI-lh'.format(area) for area in ['V2', 'V3', 'V4']] +
            ['R_{}_ROI-rh'.format(area) for area in ['V2', 'V3', 'V4']]
        ),
        'HCPMMP1_visual_ventral': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1']]
        ),
        'HCPMMP1_visual_lateral': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'V3CD', 'LO1', 'LO2', 'LO3', 'V4t', 'FST', 'MT', 'MST', 'PH']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'V3CD', 'LO1', 'LO2', 'LO3', 'V4t', 'FST', 'MT', 'MST', 'PH']]
        ),
        'HCPMMP1_somato_sens_motor': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '4', '3a', '3b', '1', '2']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '4', '3a', '3b', '1', '2']]
        ),
        'HCPMMP1_paracentral_midcingulate': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '24dd', '24dv', '6mp', '6ma', '5m', '5L', '5mv', '33pr', 'p24pr']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '24dd', '24dv', '6mp', '6ma', '5m', '5L', '5mv', '33pr', 'p24pr']]
        ),
        'HCPMMP1_premotor': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '55b', '6d', '6a', 'FEF', '6v', '6r', 'PEF']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '55b', '6d', '6a', 'FEF', '6v', '6r', 'PEF']]
        ),
        'HCPMMP1_pos_opercular': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '43', 'FOP1', 'OP4', 'OP1', 'OP2-3', 'PFcm']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '43', 'FOP1', 'OP4', 'OP1', 'OP2-3', 'PFcm']]
        ),
        'HCPMMP1_audiotory_early': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'A1', 'LBelt', 'MBelt', 'PBelt', 'RI']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'A1', 'LBelt', 'MBelt', 'PBelt', 'RI']]
        ),
        'HCPMMP1_audiotory_association': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'A4', 'A5', 'STSdp', 'STSda', 'STSvp', 'STSva', 'STGa', 'TA2']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'A4', 'A5', 'STSdp', 'STSda', 'STSvp', 'STSva', 'STGa', 'TA2']]
        ),
        'HCPMMP1_insular_front_opercular': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3', 'MI', 'AVI', 'AAIC', 'Pir', 'FOP4', 'FOP5']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '52', 'PI', 'Ig', 'PoI1', 'PoI2', 'FOP2', 'FOP3', 'MI', 'AVI', 'AAIC', 'Pir', 'FOP4', 'FOP5']]
        ),
        'HCPMMP1_temporal_med': (
            ['L_{}_ROI-lh'.format(area) for area in ['H', 'PreS', 'EC', 'PeEc', 'PHA1', 'PHA2', 'PHA3']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'H', 'PreS', 'EC', 'PeEc', 'PHA1', 'PHA2', 'PHA3']]
        ),
        'HCPMMP1_temporal_lat': (
            ['L_{}_ROI-lh'.format(area) for area in ['PHT', 'TE1p', 'TE1m', 'TE1a', 'TE2p', 'TE2a', 'TGv', 'TGd', 'TF']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'PHT', 'TE1p', 'TE1m', 'TE1a', 'TE2p', 'TE2a', 'TGv', 'TGd', 'TF']]
        ),
        'HCPMMP1_temp_par_occ_junction': (
            ['L_{}_ROI-lh'.format(area) for area in ['TPOJ1', 'TPOJ2', 'TPOJ3', 'STV', 'PSL']] +
            ['R_{}_ROI-rh'.format(area)
             for area in ['TPOJ1', 'TPOJ2', 'TPOJ3', 'STV', 'PSL']]
        ),
        'HCPMMP1_partietal_sup': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'LIPv', 'LIPd', 'VIP', 'AIP', 'MIP', '7PC', '7AL', '7Am', '7PL', '7Pm']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'LIPv', 'LIPd', 'VIP', 'AIP', 'MIP', '7PC', '7AL', '7Am', '7PL', '7Pm']]),
        'HCPMMP1_partietal_inf': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'PGp', 'PGs', 'PGi', 'PFm', 'PF', 'PFt', 'PFop', 'IP0', 'IP1', 'IP2']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'PGp', 'PGs', 'PGi', 'PFm', 'PF', 'PFt', 'PFop', 'IP0', 'IP1', 'IP2']]
        ),
        'HCPMMP1_cingulate_pos': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'DVT', 'ProS', 'POS1', 'POS2', 'RSC', 'v23ab', 'd23ab', '31pv', '31pd', '31a', '23d', '23c', 'PCV', '7m']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'DVT', 'ProS', 'POS1', 'POS2', 'RSC', 'v23ab', 'd23ab', '31pv', '31pd', '31a', '23d', '23c', 'PCV', '7m']]
        ),
        'HCPMMP1_frontal_orbital_polar': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '47s', '47m', 'a47r', '11l', '13l', 'a10p', 'p10p', '10pp', '10d', 'OFC', 'pOFC']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '47s', '47m', 'a47r', '11l', '13l', 'a10p', 'p10p', '10pp', '10d', 'OFC', 'pOFC']]
        ),
        'HCPMMP1_frontal_inferior': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '44', '45', 'IFJp', 'IFJa', 'IFSp', 'IFSa', '47l', 'p47r']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '44', '45', 'IFJp', 'IFJa', 'IFSp', 'IFSa', '47l', 'p47r']]
        ),
        'HCPMMP1_dlpfc': (
            ['L_{}_ROI-lh'.format(area) for area in [
                '8C', '8Av', 'i6-8', 's6-8', 'SFL', '8BL', '9p', '9a', '8Ad', 'p9-46v', 'a9-46v', '46', '9-46d']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                '8C', '8Av', 'i6-8', 's6-8', 'SFL', '8BL', '9p', '9a', '8Ad', 'p9-46v', 'a9-46v', '46', '9-46d']]
        ),
        'post_medial_frontal': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'SCEF', 'p32pr', 'a24pr', 'a32pr', 'p24']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'SCEF', 'p32pr', 'a24pr', 'a32pr', 'p24']]
        ),
        'vent_medial_frontal': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'p32', 's32', 'a24', '10v', '10r', '25']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'p32', 's32', 'a24', '10v', '10r', '25']]
        ),
        'ant_medial_frontal': (
            ['L_{}_ROI-lh'.format(area) for area in [
                'd32', '8BM', '9m']] +
            ['R_{}_ROI-rh'.format(area) for area in [
                'd32', '8BM', '9m']]
        ),
    }

    glasser_clusters['HCPMMP1_frontal_post_medial'] = glasser_clusters['post_medial_frontal']
    glasser_clusters['HCPMMP1_frontal_vent_medial'] = glasser_clusters['vent_medial_frontal']
    glasser_clusters['HCPMMP1_frontal_ant_medial'] = glasser_clusters['ant_medial_frontal']
            

    # fmt: on

    all_clusters = {}
    all_clusters.update(visual_field_clusters)
    all_clusters.update(jwg_clusters)
    all_clusters.update(glasser_clusters)
    areas = [
        item
        for sublist in [all_clusters[k] for k in all_clusters.keys()]
        for item in sublist
    ]

    return all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters


def get_all_glasser_clusters():
    # fmt: off
    labels = [
        "1", "10d", "10pp", "10r", "10v", "11l", "13l", "2", "23c", "23d",
        "24dd", "24dv", "25", "31a", "31pd", "31pv", "33pr", "3a", "3b", "4",
        "43", "44", "45", "46", "47l", "47m", "47s", "52", "55b", "5ROI",
        "5m", "5mv", "6a", "6d", "6ma", "6mp", "6r", "6v", "7AROI", "7Am",
        "7PC", "7PROI", "7Pm", "7m", "8Ad", "8Av", "8BM", "8BROI", "8C", "9-46d",
        "9a", "9m", "9p", "A1", "A4", "A5", "AAIC", "AIP", "AVI", "DVT",
        "EC", "FEF", "FFC", "FOP1", "FOP2", "FOP3", "FOP4", "FOP5", "FST", "H",
        "IFJa", "IFJp", "IFSa", "IFSp", "IP0", "IP1", "IP2", "IPS1", "Ig", "LBelt",
        "LIPd", "LIPv", "LO1", "LO2", "LO3", "MBelt", "MI", "MIP", "MST", "MT",
        "OFC", "OP1", "OP2-3", "OP4", "PBelt", "PCV", "PEF", "PF", "PFcm", "PFm",
        "PFop", "PFt", "PGi", "PGp", "PGs", "PH", "PHA1", "PHA2", "PHA3", "PHT",
        "PI", "PIT", "POS1", "POS2", "PSROI", "PeEc", "Pir", "PoI1", "PoI2", "PreS",
        "ProS", "RI", "RSC", "SCEF", "SFROI", "STGa", "STSda", "STSdp", "STSva", "STSvp",
        "STV", "TA2", "TE1a", "TE1m", "TE1p", "TE2a", "TE2p", "TF", "TGd", "TGv",
        "TPOJ1", "TPOJ2", "TPOJ3", "V1", "V2", "V3", "V3A", "V3B", "V3CD", "V4",
        "V4t", "V6", "V6A", "V7", "V8", "VIP", "VMV1", "VMV2", "VMV3", "VVC",
        "a10p", "a24", "a24pr", "a32pr", "a47r", "a9-46v", "d23ab", "d32", "i6-8", "p10p",
        "p24", "p24pr", "p32", "p32pr", "p47r", "p9-46v", "pOFC", "s32", "s6-8", "v23ab",        
    ]
    # fmt: on
    return {l:['L_{}_ROI-lh'.format(l), 'R_{}_ROI-rh'.format(l)] for l in labels}


def labels2clusters(labels, clusters=None):
    if clusters is None:
        clusters = get_clusters()[0]
    res = {}
    for cluster, names in clusters.items():
        res[cluster] = []
        for l in labels:
            if any([n in l.name for n in names]):
                res[cluster].append(l)
    return res


def rois2vertex(
    subject,
    mapping,
    hemi,
    labels,
    vmin=0,
    vmax=1,
    cmap=None,
    subjects_dir=os.environ["SUBJECTS_DIR"],
):
    """
    Return a pycortex Vertex data set that can be shown on a flatmap.
    """
    import cortex
    import numpy as np

    clusters = labels2clusters(labels)
    mask = 0.5 + np.zeros(cortex.db.fsaverage.surfaces.inflated.get()[0][0].shape[0])
    for cluster, value in mapping.items():
        labels = clusters[cluster]
        for label in labels:
            if hemi in label.name:
                mask[label.vertices] = value
    return cortex.dataset.Vertex(mask, subject, vmin=vmin, vmax=vmax, cmap=cmap)


def rois2volume(
    subject, mapping, hemi, labels, subjects_dir=os.environ["SUBJECTS_DIR"]
):
    """
    Create a volume that has labels filled with specific values.

    mapping is a dict that maps clusters to values
    """
    clusters = labels2clusters(labels)
    imgs = []
    for cluster, value in mapping.items():
        img = cluster2vol(
            cluster, clusters[cluster], subject, hemi, subjects_dir=subjects_dir
        )
        d = img.get_data()
        d[d > 0] = value
        imgs.append(img)
    img = imgs.pop()
    d = img.get_data()
    for i in imgs:
        d += i.get_data()
    return img


def cluster2vol(
    cluster, labels, subject, hemi, subjects_dir=os.environ["SUBJECTS_DIR"]
):
    """
    Convert labels in a cluster into one output label
    """
    from os.path import join
    from glob import glob
    import subprocess

    voloutfile = join(
        subjects_dir, subject, "mri", "{cluster}_label.nii".format(cluster=cluster)
    )
    labeloutfile = join(
        subjects_dir,
        subject,
        "label",
        "{hemi}.{cluster}_label.label".format(hemi=hemi, cluster=cluster),
    )
    labels = [l for l in labels if hemi in l.name]
    if not os.path.isfile(labeloutfile):
        label = labels.pop()
        if len(labels) > 0:
            _ = [label + l for l in labels]
        print(label, labeloutfile)
        label.save(labeloutfile)

    if not os.path.isfile(voloutfile):
        pass
    cluster_map, _, _, _ = get_clusters()
    labels = cluster_map[cluster]
    cmd = """
source $FREESURFER_HOME/SetUpFreeSurfer.sh;
/Applications/freesurfer/bin/mri_label2vol\\
    --regheader {subjects_dir}/{subject}/mri/orig.mgz\\
    --subject {subject}\\
    --hemi {hemi}\\
    --temp {subjects_dir}/{subject}/mri/orig.mgz\\
    --proj abs -5 5 .1\\
    --label {label}\\
    --o {outfile}
    """.format(
        subjects_dir=subjects_dir,
        subject=subject,
        hemi=hemi,
        label=labeloutfile,
        outfile=voloutfile,
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    import nibabel as nib

    return nib.load(voloutfile)


from collections import OrderedDict

areas = OrderedDict()
areas["Primary occipital"] = "vfcPrimary"
areas["Early occipital"] = "vfcEarly"
areas["Ventral occipital"] = "vfcVO"
areas["Parahippocampal"] = "vfcPHC"
areas["Temporal occipital"] = "vfcTO"
areas["Lateral occipital"] = "vfcLO"
areas["Dorsal occipital"] = "vfcV3ab"
areas["Intraparietal 1"] = "vfcIPS01"
areas["Intraparietal 2"] = "vfcIPS23"
areas["FEF"] = "vfcFEF"

areas["Anterior intraparietal sulcus"] = "JWG_aIPS"
areas["Intraparietal / postcentral sulcus"] = "JWG_IPS_PCeS"
areas["Motor cortex (hand area)"] = "JWG_M1"

areas["Posterior cingulate"] = "HCPMMP1_cingulate_pos"
areas["Paracentral / midcingulate"] = "HCPMMP1_paracentral_midcingulate"
areas["Insula"] = "HCPMMP1_insular_front_opercular"
areas["Posterior medial frontal"] = "post_medial_frontal"
areas["Anterior medial frontal"] = "ant_medial_frontal"
areas["Ventromedial frontal"] = "vent_medial_frontal"
areas["Premotor"] = "HCPMMP1_premotor"
areas["Dorsolateral prefrontal"] = "HCPMMP1_dlpfc"
areas["Ventrolateral prefrontal"] = "HCPMMP1_frontal_inferior"
areas["Orbital frontal polar"] = "HCPMMP1_frontal_orbital_polar"


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
