import pandas as pd
import numpy as np


def get_per_sample_data(data, meta, latency):
    cvals = np.stack(meta.contrast_probe)
    samples = latency + np.arange(0, 1, 0.1)
    dt = np.unique(data.time)
    id_t = []
    for d in samples:
        id_t.append(dt[np.argmin(np.abs(dt - d))])
    data = data.loc[data.time.isin(id_t), :]
    data.loc[:, 'sample'] = data.loc[:, 'time'].replace(
        {d: i for i, d in enumerate(id_t)}).astype(int)

    data.set_index(['trial', 'sample'], inplace=True)
    dm = pd.DataFrame(cvals, index=meta.index.values,
                      columns=np.arange(10)).stack()
    dm.index.names = ['trial', 'sample']
    dm.name = 'contrast'
    return data.join(dm)
