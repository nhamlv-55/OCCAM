#!/usr/bin/env bash

NO_OF_ITERATION=1
NO_OF_SAMPLING=2
WORKDIR="/Users/e32851/workspace/OCCAM/examples/portfolio/tree/"
CURRENT_DIR=${PWD}
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -i|--iter)
	          NO_OF_ITERATION="$2"
	          shift # past argument
	          shift # past value
	          ;;
        -folder|--folder)
	          WORKDIR="$2"
	          shift # past argument
	          shift # past value
            ;;
        -s|--sampling)
            NO_OF_SAMPLING="$2"
	          shift # past argument
	          shift # past value
	          ;;
    esac
done

echo Running training with $NO_OF_ITERATION iterations, $NO_OF_SAMPLING sampling each iteration
echo at $WORKDIR

#bootstrap
python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py

cd $WORKDIR

for i in $( seq 0 $NO_OF_ITERATION)
do
         echo Iteration $i
         seq -w -s "_${i}\n" 0 $NO_OF_SAMPLING | parallel ./build.sh -folder {}
         python3 /Users/e32851/workspace/OCCAM/razor/MLPolicy/main.py
done

cd $CURRENT_DIR