#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#CORPUS=/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s1/all_sentence.txt
#VOCAB_FILE=/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s1/vocab.txt
#COOCCURRENCE_FILE=/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s1/cooccurrence.bin
#COOCCURRENCE_SHUF_FILE=/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s1/cooccurrence.shuf.bin
#BUILDDIR=build
#SAVE_FILE=/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s1/vectorsi

#CORPUS=/hdd/lujunyu/dataset/meituan-sa/all_cut_sent.txt
CORPUS=/hdd/lujunyu/dataset/meituan-sa/all_cut_sent.txt
VOCAB_FILE=/hdd/lujunyu/dataset/meituan/glove/vocab.txt
COOCCURRENCE_FILE=/hdd/lujunyu/dataset/meituan/glove/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=/hdd/lujunyu/dataset/meituan/glove/cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=/hdd/lujunyu/dataset/meituan/glove/vectors

VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=200
MAX_ITER=100
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=10

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ python eval/python/evaluate.py"
       python eval/python/evaluate.py
   fi
fi
