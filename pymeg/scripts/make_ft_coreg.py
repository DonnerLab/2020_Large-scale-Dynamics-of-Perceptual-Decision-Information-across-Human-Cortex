#!/usr/bin/env python
"""
Script for initializing pymeg make_trans function for making transformation matrix
"""
import argparse
from pymeg.read_fieldtrip import get_info_for_epochs, read_ft_epochs
from pymeg import source_reconstruction as sr
import os


import contextlib
import os


# https://stackoverflow.com/questions/2059482/python-temporarily-modify-the-current-processs-environment
@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Fieldtrip epochs file",
                        default=None, type=str)
    parser.add_argument("--raw", help="Raw folder that the epochs come from",
                        default=None, type=str)
    parser.add_argument("--output", help="Transformation file name (e.g. S01-01-trans.fif",
                        default=None, type=str)
    parser.add_argument("--subject", help="Subject for which this transformation matrix is",
                        default=None, type=str)
    parser.add_argument("--subject_dir", help="Freesurfer subject dir, defaults to env var",
                        default=None, type=str)
    parser.add_argument("--precompute", help="Precompute info necessary for making trans matrix. This allows to separate fitting transformation from reading necessary data.",
                        action='store_true')
    parser.add_argument("--cachedir", help="Set the cachedir",
                        default=None, type=str)
    parser.add_argument("--fieldtrip", help="Epochs are fieldtrip",
                        action='store_true')

    args = parser.parse_args()
    cachedir = args.cachedir
    subjectdir = args.subject_dir

    if cachedir is None:
        cachedir = os.environ['PYMEG_CACHE_DIR']

    if subjectdir is None:
        subjectdir = os.environ['SUBJECTS_DIR']

    print(cachedir, subjectdir, args.precompute)
    with modified_environ(PYMEG_CACHE_DIR=cachedir, SUBJECTS_DIR=subjectdir):
        if args.precompute:
            rawinfo = get_info_for_epochs(args.raw)
            if args.fieldtrip:
                meg_data = read_ft_epochs(args.epochs, rawinfo,
                                          cachedir=os.environ[
                                              'PYMEG_CACHE_DIR'],
                                          trialinfo_col=-1)
                print('Converted FT epochs to: %s' % meg_data)
            else:
                meg_data = args.epochs
            trans_filename = sr.get_trans_epoch(rawinfo, meg_data)
            print('Saved trans-info in %s. ' % trans_filename)
        else:
            sr.make_trans(args.subject, args.raw, args.epochs, args.output, sdir=subjectdir)
