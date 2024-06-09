echo "Starting..."

execute_test() {
    mapfile -t lines < "$1"

    git_branch=${lines[0]}
    string=${lines[1]}

    echo " >> branch: $git_branch"
    echo " >> command: $string"

    mv $1 $1.in_progress
    git add $1.in_progress --quiet
    git commit -m "Running $1" --quiet
    git push --quiet

    # CHECKOUT TO TEST BRANCH AND RUN TEST
    git checkout $git_branch --quiet
    git reset --hard --quiet HEAD
    $string &> "$1.logs"
    
    # PUSH TEST RESULTS TO TEST BRANCH
    git add -A
    git commit -m --quiet "Test results"
    git push --quiet

    # RETURN TO TRAIN_QUEUE BRANCH TO MOVE TEST STATUS TO DONE AND CONTINUE POLLING
    git checkout train_queue --quiet
    git reset --hard --quiet HEAD
    mv $1.in_progress $1.done
    git add --quiet $1.done
    git commit --quiet -m "Finished $1"
    git push --quiet
}

while true; do
    git pull --quiet
    for file in *.test; do
        if [[ -e $file ]]; then
            echo "Test found. Executing..."
            execute_test $file
        fi
    done
    sleep 60
done