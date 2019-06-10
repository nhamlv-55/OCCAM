#!/usr/bin/env bash
NO_OF_ITERATION=1
NO_OF_SAMPLING=2
WORKDIR="/Users/e32851/workspace/OCCAM/examples/portfolio/tree/"
while getopts iter:sampling:workdir option
do
    case "${option}"
    in
        iter) NO_OF_ITERATION=${OPTARG};;
        sampling) NO_OF_SAMPLING=${OPTARG};;
        workdir) WORKDIR=${OPTARG};;
    esac
done
echo Running training with $NO_OF_ITERATION iterations, $NO_OF_SAMPLING sampling each iteration
echo at $WORKDIR

#bootstrap
python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py

for i in 0 .. $NO_OF_ITERATION
do
         echo Iteration $i
         (cd $WORKDIR ;   seq -w 0 $NO_OF_SAMPLING | parallel ./build.sh -folder {})
         python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py
done
