#!/bin/bash

NOISEFILE=true



## Input Handling
if [ -z "$1" ] || [ -z "$2" ] ; then
		echo "Usage: $0 <VTHRESH> <NUMFEATURES>"
		exit
fi

VTHRESH=$1
NF=$2

RE='^[0-9]+([.][0-9]+)?$'

if ! [[  $VTHRESH =~ $RE ]] ; then
		echo "Error: VThresh must be a number."
		exit
fi

if ! [[ $NF =~ $RE ]] ; then
		echo "Error: NumFeatures must be a number."
		exit
fi

echo 

## General Variables
EXEC=/home/jcarroll/runs/run
LOGDIR=./log
LOGFILE=$LOGDIR/log
LUA_FILE=`find *.lua`
PARAM="${LUA_FILE%.*}.params"
OUTPUT=./output/


## Parallelization Stuff
BATCHWIDTH=2
COLUMNS=2
ROWS=2
NRANKS=$(($BATCHWIDTH * $COLUMNS * $ROWS))
NTHREADS=2
export OMP_NUM_THREADS=$NTHREADS
MPI_RUN="mpirun -np $NRANKS"

## Generating Parameter File
LUACHECK=`lua $LUA_FILE $VTHRESH $NF`

if [ `echo $LUACHECK | awk '{print $1}'` == "ERROR" ]; then
		echo "Error in param file:"
		echo $LUACHECK
		exit
fi
echo "Generating params file: lua $LUA_FILE $VTHRESH $NF > $PARAM"
echo

echo "$LUACHECK" > $PARAM


## Checking for output and log directories
if [ ! -d "$OUTPUT" ] || [ ! -d "$OUTPUT" ]; then
		if [ ! -d "$OUTPUT" ]; then
				echo "Output directory not found, creating at $OUTPUT"
				mkdir "$OUTPUT"
		fi 
		
		if [ ! -d "$LOGDIR" ]; then
				echo "Log directory not found, creating at $LOGDIR"
				mkdir "$LOGDIR"
		fi 
		echo
fi

## Putting together command to run
COMMAND="$MPI_RUN $EXEC -p $PARAM -batchwidth $BATCHWIDTH -l $LOGFILE -rows $ROWS -columns $COLUMNS -t $NTHREADS"


DICTIONARY_FILE=`awk '/initWeightsFile/{ print $3 }' $PARAM | tr -d \" | tr -d \;`
if [[ $NOISEFILE  ]] && [[ ! -a $DICTIONARY_FILE ]]; then
		echo "ERROR: $DICTIONARY_FILE does not exist."
		exit
fi

echo "--------------------------------"
echo "Running PetaVision with CIFAR-10"
echo "--------------------------------"
echo "    Node              = `hostname`"
echo "    Executable        = $EXEC"
echo "    Log File          = $LOGFILE"
echo "    Param file        = $PARAM"
echo "    VThresh           = $VTHRESH"
echo "    Num Features      = $NF"
if [[ $NOISEFILE ]] ; then
echo "    Dictonary File    = $DICTIONARY_FILE"
fi
echo "--------------------------------"
echo "Executing: $COMMAND"
echo 
$COMMAND &
