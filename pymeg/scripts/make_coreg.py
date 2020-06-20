#!/usr/bin/env python
"""
Script for initializing pymeg make_trans function for making transformation matrix
"""
import argparse
from pymeg.read_fieldtrip import get_info_for_epochs, read_ft_epochs
from pymeg import source_reconstruction as sr
import os


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

    args = parser.parse_args()

    rawinfo = get_info_for_epochs(args.raw)
    meg_data = read_ft_epochs(args.epochs, rawinfo, cachedir=os.environ['PYMEG_CACHE_DIR'],
                              trialinfo_col=-1)

    sr.make_trans(args.subject, args.raw, args.epochs, args.output)
