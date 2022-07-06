#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""
A tool to check includes against linked libraries.
"""

import argparse
import logging
import os
import sys
import glob
import re


def all_lib_files(lib, base_path):
    for source in glob.iglob(f"{base_path}/lib/{lib}/**/*.*pp", recursive=True):
        yield source
    for header in glob.iglob(f"{base_path}/include/{lib}/**/*.*pp", recursive=True):
        yield header


def get_include_deps_for_lib(lib, files):
    """
    Return dict containing dict of dependencies,
    with values as the files with that dependency.
    e.g.
    return {
        poplibs_support {
            file_in_popops_depending_on_poplibs_support.cpp
        },
        poputil {
            file_in_popops_depending_on_poputil.cpp
        }
    }
    """
    deps = dict()
    for file_name in files:
        with open(file_name) as file:
            current_file_deps = set()
            for line in file:
                match = re.search(r"#\s*include\s+[<,\"](pop.*?)/", line)
                if match:
                    dep = match.group(1)
                    if dep != lib:
                        current_file_deps.add(match.group(1))
            for dep in current_file_deps:
                deps.setdefault(dep, []).append(file_name)
    return deps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "SRC_DIR",
        metavar="poplibs-src-dir",
        help="Root of poplibs src dir",
        type=str,
    )
    parser.add_argument(
        "-d",
        help="A library followed by it's dependencies",
        type=str,
        nargs="+",
        action="append",
        required=True,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("INCLUDE_DEPENDANCY_CHECK_LOG_LEVEL", "INFO"),
        format="[%(levelname)s] %(asctime)s: %(message)s",
    )

    declared_dependencies = {l[0]: l[1:] for l in args.d}
    libs = declared_dependencies.keys()
    include_dependencies = {lib: get_include_deps_for_lib(
        lib, all_lib_files(lib, args.SRC_DIR)) for lib in libs}

    error = False  # Run through all libraries to get full printout before exiting

    # See if we are over declaring linkage
    for lib in libs:
        for dep in declared_dependencies[lib]:
            # popsolver is part of gccs
            if dep not in include_dependencies[lib] and dep.startswith("pop") and dep != "popsolver":
                logging.warn(
                    f"Library {lib} links {dep}, but appears never to use it")
                error = True

    # See if we are under declaring linkage
    for lib in libs:
        for dep in include_dependencies[lib]:
            if dep not in declared_dependencies[lib]:
                logging.error(
                    f"Found dependency {lib} -> {dep} not present in declared link libraries ({declared_dependencies[lib]})\n"
                    f"{include_dependencies[lib][dep]}")
                error = True
    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()
