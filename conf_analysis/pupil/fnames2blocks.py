from . import load_edfs as le
from pylab import *
import time
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

print('sb2fname = {')
for subject in ['S%02i'%i for i in range(1, 16)]:
    edf, mf, s = le.listfiles('/home/nwilming/conf_data/%s'%subject)

    edf = dict((list(edf.keys())[k], edf[list(edf.keys())[k]]) for k in argsort([time.mktime(k) for k in list(edf.keys())]))
    mf = dict((list(mf.keys())[k], mf[list(mf.keys())[k]]) for k in argsort([time.mktime(k) for k in list(mf.keys())]))
    d = {}
    mfkeys = list(mf.keys())
    edfkeys = list(edf.keys())
    # For each day list files sorted by time
    unique_days = unique([int('%s%s%s'%(t.tm_year, t.tm_mon, t.tm_mday)) for t in edfkeys])

    days2sessions = dict((d, i+1) for i, d in enumerate(sorted(unique_days)))

    ordered_edf = [edfkeys[k] for k in argsort([time.mktime(a) for a in edfkeys])]
    ordered_mf = [mfkeys[k] for k in argsort([time.mktime(b) for b in mfkeys])]

    assert len(ordered_edf) == len(ordered_mf)

    # Check that time order between consecutive edfs and mfs and edf and mf is correct
    for i in range(len(ordered_edf)-1):
        if not time.mktime(ordered_edf[i]) < time.mktime(ordered_edf[i+1]):
            raise RuntimeErrro('EDF order not correct: %s'%edf[ordered_edf[i]])
        if not time.mktime(ordered_mf[i]) < time.mktime(ordered_mf[i+1]):
            raise RuntimeError('EDF order not correct: %s'%edf[ordered_mf[i]])
        if not time.mktime(ordered_edf[i]) < time.mktime(ordered_mf[i]):
            raise RuntimeError('EDF after Matlab:', time.mktime(ordered_edf[i]), time.mktime(ordered_mf[i]))


    for e, m in zip(ordered_edf, ordered_mf):
        day = int('%s%s%s'%(e.tm_year, e.tm_mon, e.tm_mday))
        session = days2sessions[day]
        if not session in list(d.keys()):
            d[session] = [(edf[e], mf[m])]
        else:
            d[session].append((edf[e], mf[m]))

    print("\t'%s':{"%(subject))
    for session, files in d.items():
        print('\t\t%i:{'%session)
        for block, f in enumerate(files):
            print('\t\t\t%i:%s,'%(block, f))
        print('\t\t},')
    print('\t},')
print('}')

print('\n')
print('''
fnames2sb = {}
for sub, d in sb2fname.iteritems():
    fnames2sb[sub] = dict((v[0].split('/')[-1], (session, block, v[1])) for session, bvs in d.iteritems() for block, v in bvs.iteritems())
''')
