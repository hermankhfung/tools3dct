#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Extract x- and y-displacements from transformation output in raw format

Called as part of unwarp_slurm_array.sh.
'''

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract x and y mapping from bUnwarpJ raw transformation file')
    parser.add_argument('--input', dest='rootnames', nargs='+', metavar='STRING', help='root name of files')

    args = parser.parse_args()
    
    for rootname in args.rootnames:
        f = open(rootname+'_rawtransf.txt', 'r')
        x_file = open(rootname+'_mapping_x.txt', 'w')
        y_file = open(rootname+'_mapping_y.txt', 'w')
        for line in f:
            if "X Trans" in line:
                line = next(f,'').rstrip()
                while line != '':
                    x_file.write(line+'\n')
                    line = next(f,'').rstrip()
            if "Y Trans" in line:
                line = next(f).rstrip()
                while line != '':
                    y_file.write(line+'\n')
                    line = next(f,'').rstrip()
        f.close()
        x_file.close()
        y_file.close()
