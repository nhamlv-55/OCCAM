#!/usr/bin/env bash

# Make sure we exit if there is a failure
set -e

#check that the require dependencies are built
declare -a bitcode=("hmmer.bc")

for bc in "${bitcode[@]}"
do
    if [ -a  "$bc" ]
    then
        echo "Found $bc"
    else
        echo "Error: $bc not found. Try \"make\"."
        exit 1
    fi
done

export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash/occam.log

rm -rf slash hmmer_slashed

# Build the manifest file
cat > hmmer.manifest <<EOF
{ "main" : "hmmer.bc"
, "binary"  : "hmmer_slashed"
, "modules"    : []
, "native_libs" : []
, "name"    : "hmmer"
}
EOF


# Run OCCAM
cp ./hmmer ./hmmer_orig
slash --stats --devirt --work-dir=slash hmmer.manifest
cp ./slash/hmmer_slashed .
