#!/usr/bin/env bash
WORKDIR="/Users/e32851/workspace/OCCAM/examples/portfolio/tree/"
NO_OF_ITERATION=2
NO_OF_SAMPLING=2

for i in 0 .. $NO_OF_ITERATION
do
         echo Iteration $i
         (cd $WORKDIR ; parallel ./build.sh -folder run{} ::: {0..$NO_OF_SAMPLING} )
         python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py
done
