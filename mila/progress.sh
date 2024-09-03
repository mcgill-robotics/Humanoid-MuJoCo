# check if the progress file exists
if [ ! -f data/SAC/train.progress ]; then
    echo "No progress file data/SAC/train.progress yet."
fi

watch -n 0.1 cat data/SAC/train.progress