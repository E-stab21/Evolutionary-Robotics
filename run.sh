# Run Python script limited to 40% CPU
PYTHONPATH=$(pwd)
export PYTHONPATH
systemd-inhibit --why="Running Evolutionary Robotics Experiment" --what=idle:sleep \
systemd-run --scope -p CPUQuota=100% python /home/ethan/Projects/Evolutionary-Robotics/src/experiment.py