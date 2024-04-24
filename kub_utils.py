import os
import uuid
import time
import logging
import datetime
import requests
import traceback
from pathlib import Path
from copy import deepcopy

from kubernetes import client, config, watch
import json
from kub_config import *
# from opt_config import batch_split
# from opt_config import METADATA_TEMPLATE

def CreateMetaData(point, tag):
    metadata = deepcopy(METADATA_TEMPLATE)
    metadata['user'].update([
        ('tag', tag),
        ('params', str(point)),
    ])
    return json.dumps(metadata)

def status_checker(api, job) -> str:
    api_response = api.read_namespaced_job_status()
    for pod in api_response.items:
        status = pod.status.phase
        container_status = pod.status.container_statuses[0]
        if container_status.started is False or container_status.ready is False:
            waiting_state = container_status.state.waiting
            if waiting_state.message is not None and 'Error' in waiting_state.message:
                status = waiting_state.reason
        print(pod.metadata.name + " " + status)

    active = job.obj['status'].get('active', 0)
    succeeded = job.obj['status'].get('succeeded', 0)
    failed = job.obj['status'].get('failed', 0)
    if succeeded:
        return 'succeeded'
    elif active:
        return 'wait'
    elif failed:
        return 'failed'
    return 'wait'


def get_experiment_folder() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def job_status(jobs_status):
    if 'failed' in jobs_status:
        return 'failed'
    elif all([status == 'succeeded' for status in jobs_status]):
        return 'exited'
    return 'wait'


def to_kube_env(envs) -> list:
    kube_env = []
    for k, v in envs.items():
        kube_env.append({"name": str(k), "value": str(v)})
    return kube_env

def run_kube_job_new(api,
                 job_spec: dict,
                 envs: dict,
                 job_folder: str) -> str:
    job_tag = "-".join(job_folder.split("/")[-2:])
    job_uuid: str = f"ek-{str(uuid.uuid4())[:5]}-{job_tag}"
    job_spec["metadata"]["name"] = job_spec["metadata"]["name"].format(job_uuid)

    job_spec["spec"]["template"]["spec"]["containers"][0]["env"] = to_kube_env(envs)
    logging.basicConfig(level=logging.INFO)
    job = api.create_namespaced_job('ekurbatov', job_spec)
    logging.info(f"JOB: {job_uuid} was started. Tag is {job_tag}")
    return job


def run_batch(metaData, api, rFlag=False):

    paramsM = str(json.loads(metaData)['user']['params'][1:-1])
    print(paramsM.__class__, paramsM)
    logging.basicConfig(level=logging.INFO)
    config.load_kube_config(config_file='~/.kube/config')
    batch_size = batch_split
    AZURE_DATA_URI = "/output/"
    baseName = str(json.loads(metaData)['user']['tag'])
    procs = []
    nEvents_in = 500000#100000#500000  # 485879
    n = nEvents_in
    k = batch_size
    startPoints = [i * (n // k) + min(i, n % k) for i in range(k)]
    chunkLength = [(n // k) + (1 if i < (n % k) else 0) for i in range(k)]
    chunkLength[-1] = chunkLength[-1] - 1
    if rFlag:
        nFolders = len(os.listdir(str(Path(HOST_LOCALOUTPUT_DIRECTORY) / baseName)))
    for jobN in range(batch_size):
        if rFlag:
            # nFolders = len(os.listdir(str(Path(HOST_LOCALOUTPUT_DIRECTORY) / baseName)))
            job_folder = str(Path(HOST_OUTPUT_DIRECTORY) / baseName / str(
                jobN + nFolders+1))
            local_job_folder = str(Path(HOST_LOCALOUTPUT_DIRECTORY) / baseName / str(
                jobN + nFolders+1))
        else:
            job_folder = str(Path(HOST_OUTPUT_DIRECTORY) / baseName / str(jobN))
            local_job_folder = str(Path(HOST_LOCALOUTPUT_DIRECTORY) / baseName / str(jobN))
        envs = {
            "first_event": startPoints[jobN],
            "nEvents": chunkLength[jobN],
            "jName": baseName,
            "jNumber": jobN + 1,
            "sFactor": 1,
            "AZURE_OUTPUT_DATA_URI": os.path.join(AZURE_DATA_URI, job_folder),
            "PARAMS": str(json.loads(metaData)['user']['params'][1:-1])}
        print(envs)
        job_spec = deepcopy(JOB_SPEC)
        proc = run_kube_job_new(api, job_spec, envs, local_job_folder)
        procs.append(proc)
    return {'jobs': procs, 'metadata': metaData, 'path': str(Path(HOST_LOCALOUTPUT_DIRECTORY) / baseName), 'start': datetime.datetime.now()}


def SubmitKubJobs(point, tag, api, repeat_flag=False):
    return run_batch(CreateMetaData(point, tag), api, rFlag=repeat_flag)
