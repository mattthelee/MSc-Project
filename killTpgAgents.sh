#!/bin/bash
# Kills all running TPGAgent processes currently running by this user
user=$(whoami)
processes=$(ps -aux | grep $user | grep -E "tpgAgent" | grep -v "grep" | awk '{print $2}')
for process in $processes; do
        echo "Killing tpgAgent process: $process"
        kill $process
done
processes=$(ps -aux | grep $user | grep -E "minecraft|xinit|Minecraft" | grep -v "grep" | awk '{print $2}')
for process in $processes; do
        echo "Killing Minecraft process: $process"
        kill $process
done
