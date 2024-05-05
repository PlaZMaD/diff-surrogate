import pickle
import json
import numpy as np
import time
# from commons import FCN,  StripFixedParams, AddFixedParams, ParseParams
import copy
# from opt_config import METADATA_TEMPLATE, SLEEP_TIME
import os
# from run_kub import *
from sklearn.ensemble import GradientBoostingRegressor

from kub_config import *
import uproot4 as uproot
from os.path import isfile, join
from copy import deepcopy
import awkward1 as ak
from pathlib import Path
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_params_from_json(jsonFile):
    with open(jsonFile) as result_file:
        result = json.load(result_file)
        params = np.array(result['params'])
        params = [float(p) for p in params]
        W = float(result['w'])
        W_sc = float(result['w_sc'])
        return params, W, W_sc

def load_data(dirs, branches, treeName='cbmsim', lazy=False, lLibrary='pd', lCut=None):
    dataBase = {'fPdgCode': uproot.AsJagged(uproot.AsDtype('>i4')),
                'fX': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fY': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fZ': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fPx': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fPy': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fPz': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fW': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fStartX': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fStartY': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fStartZ': uproot.AsJagged(uproot.AsDtype('>f4')),
                'fTrackID': uproot.AsJagged(uproot.AsDtype('>i4')),
                'fEventId': uproot.AsJagged(uproot.AsDtype('>i4')),
                'fDetectorID': uproot.AsJagged(uproot.AsDtype('>i4')),
                'fLength': uproot.AsJagged(uproot.AsDtype('>f4'))
                }

    if lazy:
        print("lazy is not implemented")
        return 0
    outputs = {}
    outputs_keys = []
    for folder in dirs:
        lFile = folder  # join(folder, filename)
        if isfile(lFile):
            key = os.path.dirname(folder)  # basename(os.path.dirname(folder))
            nRepeats = 1
            while key in outputs_keys:
                nRepeats = nRepeats + 1
                key = key + "_" + str(nRepeats)
            outputs_keys.append(key)
            try:
                lTree = uproot.open(lFile)
                # print("Open file: {}".format(lFile))
                if any([treeName in lName for lName in lTree.keys()]):
                    lTree = lTree[treeName]
                else:
                    print("Bad file {}".format(lFile))
                    print(lTree.keys())
                    continue
                outputs[key] = {}
                # branchdict = {}
                for branch in branches.keys():
                    if branch == 'MCEventHeader.':
                        outputs[key][branch] = lTree[branch].arrays(
                            ['MCEventHeader.{}'.format(item) for item in branches[branch]],
                            cut=lCut[branch] if lCut else None, library=lLibrary, how=dict)
                    else:
                        outputs[key][branch] = lTree[branch].arrays(
                            {'{}.{}'.format(branch, item): dataBase[item] for item in branches[branch]},
                            cut=lCut[branch] if lCut else None, library=lLibrary, how=dict)
                        # print(key, branch, len(outputs[key][branch]['{}.fPdgCode'.format(branch)]))
            except OSError:
                print("bad file {}".format(lFile))
                continue
        else:
            print("no such file ", join(folder, lFile))
    return outputs




def calc_FCNs(dirName, fileName, nFiels, weight, sc_weight, tfilter=True, acceptance_limit=(200, 300)):
    with_snd = True
    basePath = sorted(Path(dirName).glob(f'*/{fileName}'))

    assert len(basePath) % nFiels == 0

    branches_to_load_base = ['fluxDetPoint']
    snd_planes = 'TTPoint'
    if with_snd:
        branches_to_load_base.append(snd_planes)
    branches_to_load = {}
    if tfilter:
        branches_to_load_base.append('strawtubesPoint')

    branches_to_load_base.append('vetoPoint')
    branches_to_load_base = {
        i: [k for k in ['fTrackID', 'fPdgCode', 'fX', 'fY', 'fZ', 'fDetectorID', 'fPx', 'fPy', 'fPz', 'fLength']]
        for i in branches_to_load_base}
    branches_to_load['MCTrack'] = ['fPdgCode', 'fW']
    branches_to_load['MCEventHeader.'] = ['fEventId']

    load_branches = deepcopy(branches_to_load)
    load_branches.update(branches_to_load_base)
    # ic(deepcopy(branches_to_load).update(branches_to_load_base))
    base = load_data(basePath, load_branches, lCut=None, lLibrary="ak")
    for key, data in base.items():
        lPath = key.split('/')
        evKey = (int(lPath[-1]) + 1) * 100000000
        data['MCEventHeader.']['eventID'] = data['MCEventHeader.']['MCEventHeader.fEventId'] + evKey

    tKey = next(iter(base.keys()))
    root_data = {lkey: {rkey: ak.concatenate([base[i][lkey][rkey] for i in base.keys()], axis=0) for rkey in
                        base[tKey][lkey].keys()} for lkey in base[tKey].keys()}

    for key in branches_to_load_base:
        root_data[key]['event_id'] = ak.broadcast_arrays(root_data['MCEventHeader.']['eventID'],
                                                         ak.zeros_like(root_data[key]['{}.fPdgCode'.format(key)]))[0]
        if 'MCTrack' not in key:
            root_data[key]['W'] = root_data['MCTrack']['MCTrack.fW'][root_data[key]['{}.fTrackID'.format(key)]]
    mu_only_data = {lKey: {key: root_data[lKey][key][abs(root_data[lKey]['{}.fPdgCode'.format(lKey)]) == 13] for key in
                           root_data[lKey].keys()} for lKey in branches_to_load_base}
    white_list = None
    if tfilter:
        tData = mu_only_data['strawtubesPoint']
        tflat = pd.DataFrame({key: ak.flatten(tData[key], axis=None) for key in tData.keys()})
        events_white_list = {}
        for i in range(4):
            bName = f"T{4 - i}"
            bFlat = tflat[tflat['strawtubesPoint.fDetectorID'] >= (4 - i) * 1e7]
            tflat = tflat[tflat['strawtubesPoint.fDetectorID'] < (4 - i) * 1e7]
            events_white_list[bName] = pd.unique(bFlat['event_id'])

        filter_mask = np.isin(events_white_list['T1'], events_white_list['T4'])
        white_list = events_white_list['T1'][filter_mask]
        t2t3_events = np.unique(np.concatenate(
            (events_white_list['T1'], events_white_list['T2']), axis=0))
        filter_mask_23 = np.isin(white_list, t2t3_events)
        white_list = white_list[filter_mask_23]

    vflat = None

    vData = mu_only_data['vetoPoint']
    vflat = pd.DataFrame({key: ak.flatten(vData[key], axis=None) for key in vData.keys()})
    vflat.rename(columns={col: col.split('.')[-1] for col in vflat.columns}, inplace=True)
    vflat['P'] = np.sqrt(np.square(vflat['fPx']) + np.square(vflat['fPy']) + np.square(vflat['fPz']))

    data = mu_only_data['fluxDetPoint']
    flat = {key: ak.flatten(data[key], axis=None) for key in data.keys()}
    dfFlat = pd.DataFrame(flat)
    reduced = dfFlat.groupby(['event_id', '{}.fTrackID'.format('fluxDetPoint')], as_index=False).mean()
    reduced.rename(columns={col: col.split('.')[-1] for col in reduced.columns}, inplace=True)
    reduced['P'] = np.sqrt(np.square(reduced['fPx']) + np.square(reduced['fPy']) + np.square(reduced['fPz']))
    reduced['Pt'] = np.sqrt(np.square(reduced['fPx']) + np.square(reduced['fPy']))
    reduced.drop(['fDetectorID', 'fPx', 'fPy', 'fPz'], axis=1, inplace=True)
    if with_snd:
        data_snd = mu_only_data[snd_planes]
        flat_snd = {key: ak.flatten(data_snd[key], axis=None) for key in data_snd.keys()}
        dfFlat_snd = pd.DataFrame(flat_snd)
        reduced_snd = dfFlat_snd.groupby(['event_id', '{}.fTrackID'.format(snd_planes)], as_index=False).mean()
        reduced_snd.rename(columns={col: col.split('.')[-1] for col in reduced_snd.columns}, inplace=True)
        reduced_snd['P'] = np.sqrt(
            np.square(reduced_snd['fPx']) + np.square(reduced_snd['fPy']) + np.square(reduced_snd['fPz']))
        reduced_snd['Pt'] = np.sqrt(np.square(reduced_snd['fPx']) + np.square(reduced_snd['fPy']))
        reduced_snd.drop(['fDetectorID', 'fPx', 'fPy', 'fPz'], axis=1, inplace=True)

    nRuns = int(len(basePath) / batch_split)
    if nRuns != 1:
        print("Strange number of runs")
    if tfilter:
        # if white_list is not None:
        reduced = reduced[reduced['event_id'].isin(white_list)]
        veto = reduced[['fX', 'fY']].values.tolist()
        kinematics = reduced[['fX','fY','fZ', 'fX','fY','fZ','fPdgCode']].values.tolist()
    return veto, kinematics

def params2opt(inParams):

    return np.array(inParams)[need2opt]

def params2sim(inParams):
    new_params = np.array(deepcopy(default_point))
    new_params[need2opt] = inParams
    return new_params.tolist()

def get_root_data():
    pass

def ProcessPoint4Server(jobs):
    try:
        X, W, W_sc = get_params_from_json(os.path.join(jobs['path'], "0", 'optimise_input.json'))
        print(X, W)
        veto, kinematics = calc_FCNs(jobs['path'], "ship.conical.MuonBack-TGeant4.root",
                             batch_split,
                             weight=W, sc_weight=W_sc, tfilter=True)
        return X, W, W_sc, veto, kinematics
    except Exception as e:
        print(e)


def load_points_from_dir(db_name='db.pkl'):
    with open (db_name, 'rb') as f:
        return pickle.load(f)

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
