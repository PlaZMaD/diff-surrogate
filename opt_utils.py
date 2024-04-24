import pickle
import json
import numpy as np
from commons import FCN,  StripFixedParams, AddFixedParams, ParseParams
import copy
from opt_config import METADATA_TEMPLATE, SLEEP_TIME
import os
from run_kub import *
from fcn_utils import *

from sklearn.ensemble import GradientBoostingRegressor
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, GradientBoostingQuantileRegressor
# from ax_configs import dbNumber
from redis_run_config import db


def StripFixedParams_multipoint(points):
    return [StripFixedParams(p) for p in points]


def ExtractParams(metadata):
    params = json.loads(metadata)['user']['params']
    return ParseParams(params)


def get_result(jobs):
    # print("get_result call!")
    results = []
    weights = np.array([])
    for i in range(len(jobs['jobs'])):
        # print(os.path.join(jobs['path'], str(i), 'optimise_input.json'))
        with open(os.path.join(jobs['path'], str(i), 'optimise_input.json')) as result_file:
          result = json.load(result_file)
          rWeights = np.array(result['kinematics'])
          weights = np.concatenate((weights, rWeights))
          results.append(result)
          print('result is: ', jobs['path'])
    # Only one job per machine calculates the weight and the length
    # -> take first we find
    weight = float(results[0]['w'])
    if weight < 3e6:
        muons_w = np.sum(weights)
    else:
        muons_w = 0
    return weight, 0, muons_w

def ProcessPoint(jobs, fcn_to_use):
    try:
        # X = ExtractParams(jobs['metadata'])
        print("""os.path.join(jobs['path'], "0", 'optimise_input.json')""")
        X, W, W_sc = get_params_from_json(os.path.join(jobs['path'], "0", 'optimise_input.json'))
        print(X, W)
        fcn_vals = calc_FCNs(jobs['path'], "ship.conical.MuonBack-TGeant4.root",
                             batch_split,
                             FCNs,
                             weight=W,
                             sc_weight=W_sc)
        myRedis = redis.Redis(
            host='localhost',
            port='6379',
            db=db['opt_db'])

        for fcn_name, fcn_val in fcn_vals.items():
            myRedis.hset(json.dumps(X, cls=NpEncoder), fcn_name, fcn_val)
        myRedis.hset(json.dumps(X, cls=NpEncoder), 'W', W)
        myRedis.hset(json.dumps(X, cls=NpEncoder), 'iteration', os.path.basename(jobs['path']))

        return X, fcn_vals[fcn_to_use]
    except Exception as e:
        print(e)


def ProcessPoint4Server(jobs):
    try:
        X, W, W_sc = get_params_from_json(os.path.join(jobs['path'], "0", 'optimise_input.json'))
        print(X, W)
        fcn_vals = calc_FCNs(jobs['path'], "ship.conical.MuonBack-TGeant4.root",
                             batch_split,
                             FCNs,
                             weight=W, sc_weight=W_sc, tfilter=True)
        return X, W, W_sc, fcn_vals
    except Exception as e:
        print(e)


def ProcessPoint_old(jobs):
    print("process Point: ", jobs)
    try:
        weight, _, muons_w = get_result(jobs)
        print('obtained weights: ', weight, muons_w)
        y = FCN(weight, muons_w, 0)
        X = ExtractParams(jobs['metadata'])
        # print(X, y)
        return X, y
    except Exception as e:
        print(e)


def ProcessJobs(jobs, tag, fcn_to_use):
    print('[{}] Processing jobs...'.format(time.time()))
    results = [ProcessPoint(point, fcn_to_use) for point in jobs]
    print(f'Got results {results}')
    results = [result for result in results if result]
    return zip(*results) if results else ([], [])

def WaitCompleteness(mpoints):
    uncompleted_jobs = mpoints
    work_time = 0
    restart_counts = 0
    while True:
        time.sleep(SLEEP_TIME)
        print(uncompleted_jobs)
        uncompleted_jobs = [any([job.is_alive() for job in jobs['jobs']]) for jobs in mpoints]

        if not any(uncompleted_jobs):
            return mpoints

        print('[{}] Waiting...'.format(time.time()))
        work_time += 60

        if work_time > 60 * 30 * 1:
            restart_counts += 1
            if restart_counts >= 3:
                print("Too many restarts")
                raise SystemExit(1)
            print("Job failed!")
            #raise SystemExit(1)
            for jobs in mpoints:
                if any([job.is_alive() for job in jobs['jobs']]):
                    [job.terminate() for job in jobs['jobs']]
                    jobs = run_batch(jobs['metadata'])
            work_time = 0


def CalculatePoints(points, tag, fcn_to_use='count'):
    tags = {json.dumps(points[i], cls=NpEncoder):str(tag)+'-'+str(i) for i in range(len(points))}
    shield_jobs = [
        SubmitKubJobs(point, tags[json.dumps(point, cls=NpEncoder)])
        for point in points
    ]
    print("submitted: \n", points)

    if shield_jobs:
        shield_jobs = WaitCompleteness(shield_jobs)
        X_new, y_new = ProcessJobs(shield_jobs, tag, fcn_to_use)
    return X_new, y_new

def load_points_from_dir(db_name='db.pkl'):
    with open (db_name, 'rb') as f:
        return pickle.load(f)

def CreateOptimizer(clf_type, space, random_state=None):
    if clf_type == 'rf':
        clf = Optimizer(
            space,
            RandomForestRegressor(n_estimators=500, max_depth=7, n_jobs=-1),
            random_state=random_state)
    elif clf_type == 'gb':
        clf = Optimizer(
            space,
            GradientBoostingQuantileRegressor(
                base_estimator=GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, loss='quantile')),
            random_state=random_state)
    elif clf_type == 'gp':
        clf = Optimizer(
            space,
            GaussianProcessRegressor(
                alpha=1e-7, normalize_y=True, noise='gaussian'),
            random_state=random_state)
    else:
        clf = Optimizer(
            space, base_estimator='dummy', random_state=random_state)

    return clf


def get_list_by_pattern(rdb, pattern):
    out = []
    cur = 0
    cur, new_keys = rdb.scan(cur, pattern)
    out = out + new_keys
    while cur != 0:
        cur, new_keys = rdb.scan(cur, pattern)
        out = out + new_keys
    return list(set(out))


def get_jobs_list_client(dirName):
    newJobsList = []
    for root, dirs, files in os.walk(dirName):
        for name in files:
            if ".json" in name:
                try:
                    with open(os.path.join(root, name)) as input:
                        newJobsList.append(json.load(input))
                except FileNotFoundError:
                    continue
    return newJobsList


def get_jobs_list(dirName):
    newJobsList = []
    for root, dirs, files in os.walk(dirName):
        for name in files:
            if ".json" in name:
                with open(os.path.join(root, name)) as input:
                    try:
                        newJobsList.append(json.load(input))
                    except json.decoder.JSONDecodeError:
                        print("bad json")
                        continue
                        #newJobsList.append()
                os.remove(os.path.join(root, name))
    return newJobsList

def complete_old_running(runPath):
    newJobsList = []
    for root, dirs, files in os.walk(runPath):
        for name in files:
            if ".json" in name:
                new_cand = {}
                try:
                    with open(os.path.join(root, name)) as input:
                        new_cand['trial'] = int(name[:-5])
                        new_cand['params'] = json.load(input)
                    os.remove(os.path.join(root, name))
                except FileNotFoundError:
                    continue
                try:
                    job_l_path = os.path.join(HOST_LOCALOUTPUT_DIRECTORY, name[:-5])
                    X, W, W_sc = get_params_from_json(os.path.join(job_l_path, "0", 'optimise_input.json'))
                    fcn_vals = calc_FCNs(job_l_path, "ship.conical.MuonBack-TGeant4.root",
                                         batch_split,
                                         FCNs,
                                         weight=W,
                                         sc_weight=W_sc)

                    new_cand['W'] = W
                    new_cand['W_sc'] = W_sc
                    new_cand['fcns'] = fcn_vals
                except Exception as e:
                    print(e)
                newJobsList.append(new_cand)
    return newJobsList
