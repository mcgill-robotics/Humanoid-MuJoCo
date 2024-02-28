## INSTRUCTIONS FOR TRAINING ON RUNPOD

- Launch GPU POD (ensure you have `bash -c "tail -f /dev/null"`) command in CMD of pod template
- Connect to pod via SSH or Web Terminal
- Run setup_shell.sh script to start a detachable shell
- Press Enter/Space when prompted
- Start your training process (here it is `python3 train.py`)
- Do `Ctrl+A+D` to detach from the shell
- You can now close the SSH terminal and/or web terminal and the process will continue running
- To reattach to the shell, connect to the pod again and run `screen -r`