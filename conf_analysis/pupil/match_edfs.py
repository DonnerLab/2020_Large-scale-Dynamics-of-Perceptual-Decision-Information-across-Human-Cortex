'''
Some of the edfs were not copied from the host PC correctly to stimulus PC. Need
to find corrupt ones and match these to those from the host PC.
'''
from . import load_edfs as le
import edfread
import locale


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
corrupt_files = {}

for sub in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12']:
        print(sub)
        edf, mf, sub = le.listfiles('/Users/nwilming/u/conf_data/'+sub)
        for _, e in edf.items():
                try:
                    preamble = edfread.read_preamble(e)
                except IOError:
                    preamble = edfread.read_preamble(e, 4)
                    rec_time = preamble.split('\n')[0].replace('** DATE: ', '')
                    corrupt_files[rec_time] = e
                #print preamble

import glob

backup_files = glob.glob('/Volumes/dump/edfs' + '*.edf')
backup_files += glob.glob('/Users/nwilming/Schreibtisch/edfs' + '*.edf')
backup_files += glob.glob('/Volumes/dump/edfs/first_batch' + '*.edf')
backup_files += glob.glob('/Volumes/dump/edfs2' + '*.edf')
backups = {}
for e in backup_files:
    preamble = edfread.read_preamble(e, 4)
    rec_time = preamble.split('\n')[0].replace('** DATE: ', '')
    backups[rec_time] = e

for t, name in corrupt_files.items():
    try:
        print(t, name, '->', backups[t])
        print('cp ', backups[t], name)
    except:
        print('No match!')

import pickle
pickle.dump(backups, open('backup_time_stamps.pickle', 'w'))
