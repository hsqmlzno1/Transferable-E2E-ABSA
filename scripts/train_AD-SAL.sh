#!/bin/sh

domains=('laptop' 'rest' 'device' 'service')
model_name='AD-SAL'

for src_domain in ${domains[@]};
do
    for tar_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
            CMD="python main.py --train --test -s ${src_domain} -t ${tar_domain} -m ${model_name} --selective | tee -a ./work/logs/AD-SAL/log.txt"
            echo Run "${CMD}"
            eval "${CMD}"
        fi
    done
done
