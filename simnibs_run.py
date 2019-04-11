# -*- coding: utf-8 -*-\
'''
    command line tool for running fem simulations of TMS and tDCS
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.

    Copyright (C) 2018  Guilherme B Saturnino, Kristoffer H Madsen, Axel Thieslcher,
    Jesper D Nielsen, Andre Antunes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import sys
import argparse
import simnibs

simnibs_version = str(simnibs.version)


def parseArguments(argv):
    parser = argparse.ArgumentParser(
        prog="simnibs", description="Prepare, run and postprocess FEM problems for SimNIBS.")
    parser.add_argument("simnibs_file", help="Input .mat or .simnibs file")
    parser.add_argument('--version', action='version', version=simnibs_version)
    parser.add_argument("--cpus", type=int, help="set number of CPU to run "
                        "simulteaneously (each takes ~6GB of RAM), only for TMS", default=1)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parseArguments(sys.argv[1:])
    simnibs.run_simulation(args.simnibs_file, args.cpus)
