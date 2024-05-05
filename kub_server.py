#from opt_utils import ProcessPoint4Server, complete_old_running
#from redis_run_config import *
import time
#from opt_config import *
import json
from opt_utils import get_list_by_pattern, get_jobs_list, ProcessPoint4Server
import os
import uuid
import time
import logging
import datetime
import requests
import traceback
from pathlib import Path
from copy import deepcopy
from multiprocessing import Process
from kubernetes import client, config
import json
from kub_config import *
from kub_utils import *
import kubernetes

def listen_jobs_json():

    json_db = []
    jobs_pool = {}
    run_dir = os.path.join(pool_dir, 'running')
    new_dir = os.path.join(pool_dir, 'new')
    completed_dir = os.path.join(pool_dir, 'completed')
    print("Loading kub configuration")

    config.load_kube_config(config_file='~/.kube/config')
    api = client.BatchV1Api()

    jobs_to_run = []
    print('Init complete, waiting for jobs')

    while True:
        print(f"Running {len(jobs_pool)} jobs out of  {max_batch_N} ")
        if len(jobs_pool) < max_batch_N:
            jobs_to_run = jobs_to_run + get_jobs_list(new_dir)
            while len(jobs_to_run) > 0 and len(jobs_pool) < max_batch_N:
                print("Starting new trial!")
                job_to_run = jobs_to_run.pop()
                trial_index = job_to_run['trial_index']
                opt_parameters = job_to_run['parameters']
                print(f"Trial {trial_index} obtained")
                parameters = opt_parameters
                jobs_pool[trial_index] = {'proc': SubmitKubJobs(parameters, str(trial_index), api), 'time': 0, 'restarts': 0,
                                          'trial': trial_index, 'parameters': opt_parameters, 'tag': job_to_run['run_tag']}
                with open(os.path.join(run_dir, f"{trial_index}.json"), 'w') as output:
                    json.dump({'parameters': opt_parameters, 'trial_index': trial_index}, output)

        completed = []
        for jobID, job in jobs_pool.items():
            if not all([api.read_namespaced_job(subJobName, 'ekurbatov').status.succeeded == 1 for subJobName in [lName.metadata.name for lName in  job['proc']['jobs']]]):
                job['time'] += SLEEP_TIME
                if job['time'] > RESTART_TIME:
                    job['restarts'] += 1
                    if job['restarts'] > MAX_RESTARTS:
                        print("Too many restarts")
                        # raise SystemExit(1)

                        out_data_r = {'trial_index': job['trial'],
                                   'parameters': [0],
                                   'W': 0,
                                   'fcns': 0,
                                    'failed':True}
                        with open(os.path.join(completed_dir, f"{job['trial']}.json"), 'w') as output:
                            json.dump(out_data_r, output)
                        os.remove(os.path.join(run_dir, f"{job['trial']}.json"))
                        completed.append(jobID)
                    for kub_job in jobs_pool[jobID]['proc']['jobs']:
                        api.delete_namespaced_job(kub_job.metadata.name, 'ekurbatov', body=kubernetes.client.V1DeleteOptions(propagation_policy='Foreground'))#propagation_policy='Foreground')
                        time.sleep(3)
                    job['time'] = 0
                    job['proc'] = SubmitKubJobs(job['parameters'], job['trial'], api)
            else:
                xwy = ProcessPoint4Server(job['proc'])
                if xwy is not None:
                    X_new, W, W_sc, veto, kinematics = xwy
                    # if y_new['tCount'] < 20 and checkStats and job['restarts'] < 5:
                    #     print("Running additional statistics")
                    #     jobs_pool[jobID] = {'proc': SubmitKubJobs(job['parameters'], str(job['trial']), api, repeat_flag=True), 'time': 0,
                    #                         'restarts': job['restarts'] + 1,
                    #                         'trial': job['trial'], 'parameters': job['parameters'], 'tag': job['tag']}
                    #     continue
                else:
                    print("RESTARTING JOB")
                    jobs_pool[jobID] = {'proc': SubmitKubJobs(job['parameters'], str(job['trial']), api), 'time': 0, 'restarts': job['restarts']+1,
                     'trial': job['trial'], 'parameters': job['parameters'], 'tag': job['tag']}
                    continue
                os.remove(os.path.join(run_dir, f"{job['trial']}.json"))
                with open(os.path.join(completed_dir, f"{job['trial']}.json"), 'w') as output:
                    out_data = {'trial_index': job['trial'],
                                'parameters': X_new,
                                'W': W,
                                'W_sc': W_sc,
                                'veto': veto,
                                'kinematics': kinematics}
                    if checkStats:
                        out_data.update({'nRuns': job['restarts']+1})
                    json.dump(out_data, output)
                    print("Saved to completed")

                json_db.append({'iteration': job['trial'],
                                'W': W,
                                'parameters': X_new,
                               'tag': job['tag']})
                with open("json_db.json", 'w') as out:
                    json.dump(json_db, out)

                completed.append(jobID)
        for jobID in completed:
            logging.info(f'Job {jobID} ready!')
            for kub_job in jobs_pool[jobID]['proc']['jobs']:
                api.delete_namespaced_job(kub_job.metadata.name, 'ekurbatov', body=kubernetes.client.V1DeleteOptions(propagation_policy='Foreground'))#propagation_policy='Foreground')
                time.sleep(3)
            jobs_pool.pop(jobID, None)
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    listen_jobs_json()
