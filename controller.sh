#!/usr/bin/env bash
WORKDIR="/Users/e32851/workspace/OCCAM/examples/portfolio/tree/"
NO_OF_ITERATION=1
NO_OF_SAMPLING=2

for i in 0 .. $NO_OF_ITERATION
do
         echo Iteration $i
         (cd $WORKDIR ;   seq -w 0 $NO_OF_ITERATION | parallel ./build.sh -folder {})
         python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py
done
