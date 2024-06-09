echo "Starting..."

execute_test() {
    mapfile -t lines < "$1"

    git_branch=${lines[0]}
    string=${lines[1]}

    echo " >> branch: $git_branch"
    echo " >> command: $string"

    mv $1 $1.in_progress
    git add $1.in_progress
    git commit -m "Running $1"
    git push

    git checkout $git_branch
    exec $string
    
    git commit -am "Test results"
    git push

    git checkout train_queue
    mv $1.in_progress $1.done
    git commit -am "Finished $1"
    git push
}

while true; do
    git pull > /dev/null
    for file in *.test; do
        if [[ -e $file ]]; then
            echo "Test found. Executing..."
            execute_test $file
        fi
    done
    sleep 60
done