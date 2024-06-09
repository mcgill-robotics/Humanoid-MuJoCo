echo "Starting..."

execute_test() {
    mapfile -t lines < "$1"

    git_branch=${lines[0]}
    string=${lines[1]}

    echo "Git branch: $git_branch"
    echo "String: $string"

    mv $1 $1.in_progress
    git commit -am "Running test $1"
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
    git pull
    for file in *.test; do
        if [[ -e $file ]]; then
            execute_test $file
        fi
    done
    sleep 60
done