#!/usr/bin/env python
# -*- coding: utf-8 -*-

# parameters of server
K8S_PROXY = 'https://cern-mc43h.ydf.yandex.net:8443'

HOST_OUTPUT_DIRECTORY = "optimization/data/gan_opt_01/"
HOST_LOCALOUTPUT_DIRECTORY = '/home/ekurbatov/shipfs/optimization/data/gan_opt_01/'
DOCKER_OUTPUT_DIRECTORY = '/output'

# HOST_SAMPLE_DIRECTORY - local folder in the cluster
HOST_SAMPLE_DIRECTORY = '/local/ship/background_2018'
DOCKER_SAMPLE_DIRECTORY = '/sample'

TIMEOUT = 60*60*20
SLEEP_TIME = 60
RESTART_TIME = 60*60
MAX_RESTARTS = 5

checkStats = False

max_batch_N = 40
batch_split = 20

# pool_dir = 'json_config'#../json_config/'
pool_dir = '/home/ekurbatov/shipfs/optimization/json_control/'

default_point = [70, 170, 0, 353.078, 125.083, 184.834, 150.193, 186.812, 40, 40, 150, 150, 2, 2, 80, 80, 150, 150, 2, 2, 72, 51, 29, 46, 10, 7, 45.6888, 45.6888, 22.1839, 22.1839, 27.0063, 16.2448, 10, 31, 35, 31, 51, 11, 24.7961, 48.7639, 8, 104.732, 15.7991, 16.7793, 3, 100, 192, 192, 2, 4.8004, 3, 100, 8, 172.729, 46.8285, 2]
need2opt = [ 4, 5, 6, 7] + [i for i in  range(38, len(default_point))]
run_tag = "first_try"

JOB_SPEC = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
        # Fill in the python script
        "name": "{}"
    },
    "spec": {
        # Don't forget about this disabled option
        # "ttlSecondsAfterFinished": 14400,
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ekship",
                        "image": "mrphys/shield_opt:sc_field_06",#"mrphys/shield_opt:snd_opt_01",#"mrphys/shield_opt:ecn3_10_11_22__wide_trench",#"mrphys/shield_opt:ecn3_140922_wide_trench",#2 - 500k, 3- 1m, 5 -500k+straw, 10 500k straw+veto+flux
                        "resources": {
                            "requests": {
                                "memory": "4Gi",
                                "cpu": "1"
                            },
                            "limits": {
                                "memory": "4Gi",
                                "cpu": "1"
                            }
                        },
                        "volumeMounts": [
                            {
                                "name": "yandex",
                                "mountPath": "/output"
                            }
                            # {
                            #     "mountPath": DOCKER_OUTPUT_DIRECTORY,
                            #     "name": "output"
                            # },
                            # {
                            #     "mountPath": DOCKER_SAMPLE_DIRECTORY,
                            #     "name": "muonsample",
                            #     # "readOnly": true
                            # }
                        ]
                    }
                ],
                "hostNetwork": True,
                "restartPolicy": "Never",
                "volumes": [
                    {
                        "name": "yandex",
                        "persistentVolumeClaim": {
                             "claimName": "ekurbatov-s3"
                        }
                    }
                    # # Use this with mount
                    # # {
                    # #     "name": "output",
                    # #     "hostPath": {
                    # #         # Fill in the python script
                    # #         "path": "",
                    # #         "type": "Directory"
                    # #     }
                    # # },
                    # # Use this with azcopy
                    # {
                    #     "name": "output",
                    #     "emptyDir": {}
                    # },
                    # {
                    #     "name": "muonsample",
                    #     "hostPath": {
                    #         "path": HOST_SAMPLE_DIRECTORY,
                    #         "type": "Directory"
                    #     }
                    # }
                ]
            }
        },
        "backoffLimit": 1
    }
}

METADATA_TEMPLATE = {
    'user': {
        'tag': '',
        'params': []
    },
    'k8s': {}
}