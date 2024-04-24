import os

import uproot4 as uproot
# import ROOT as r
import numpy as np
import json
from os.path import isfile, join
from pathlib import Path
import awkward1 as ak
import pandas as pd
import redis
from commons import FCN
from opt_config import batch_split, add_tracks_filter
from redis_run_config import db
from copy import deepcopy
from icecream import ic
from kub_config import HOST_LOCALOUTPUT_DIRECTORY

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

def get_params_from_geofile(lgeofile):
    pass
    # geofile = r.TFile(lgeofile, 'r')
    # geo = geofile.Get("FAIRGeom")
    # cave_node = geo.GetTopNode()
    # zy = {'y': [], 'z': []}
    # zx = {'x': [], 'z': []}

    # for node in cave_node.GetNodes():
    #     for subnode in node.GetNodes():
    #         nodeName = subnode.GetName()
    #         if "MagRetR" in nodeName and 'Absorb' not in nodeName:
    #             subnode.Print()
    #             zShift = subnode.GetMatrix().GetTranslation()[2]
    #             lVol =  subnode.GetVolume().GetShape()
    #             dots = lVol.GetVertices()
    #             dZ = lVol.GetDZ()
    #             vetz = [dots[i] for i in range(16)]
    #             Y = vetz[1::2]
    #             zy['z'] = zy['z'] + [-dZ+zShift, -dZ+zShift, dZ+zShift, dZ+zShift, -dZ+zShift]
    #             zy['y'] = zy['y'] + [-max(Y[:4]), max(Y[:4]), max(Y[4:]), -max(Y[4:]), -max(Y[:4])]
    #             for key, item in zy.items():
    #                 item.append(None)
    #         if "MagTopL" in nodeName and 'Absorb' not in nodeName:
    #             subnode.Print()
    #             zShift = subnode.GetMatrix().GetTranslation()[2]
    #             lVol =  subnode.GetVolume().GetShape()
    #             dots = lVol.GetVertices()
    #             dZ = lVol.GetDZ()
    #             vetz = [dots[i] for i in range(16)]
    #             X = vetz[::2]
    #             zx['z'] = zx['z'] + [-dZ+zShift, -dZ+zShift, dZ+zShift, dZ+zShift, -dZ+zShift]
    #             zx['x'] = zx['x'] + [-max(X[:4]), max(X[:4]), max(X[4:]), -max(X[4:]), -max(X[:4])]
    #             for key, item in zx.items():
    #                 item.append(None)
    # return zx, zy


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


def count_fcn(df_input, weight, momentum_limit=1, acceptance_limit=(210, 410)):
    #        #calculate the number of muons in the acceptance
    return FCN(weight,
               len((df_input.query(f"P>{momentum_limit}&abs(fX)<{acceptance_limit[0]}&abs(fY)<{acceptance_limit[1]}")).index),
               0)


def tCount_fcn(df_input, weight, momentum_limit=1, acceptance_limit=(200, 300)):
    return len((df_input.query(f"P>{momentum_limit}&abs(fX)<{acceptance_limit[0]}&abs(fY)<{acceptance_limit[1]}")).index)


def veto_fcn(df_input, weight):
    prData = df_input.copy()
    prData['fcn'] = prData['fLength']/prData['P']
    return FCN(weight, prData['fcn'].sum(), 0)


def sx_fcn(df_input, weight, momentum_limit=1, acceptance_limit=(200, 400)):
    lData = df_input.query(f"P>{momentum_limit}&abs(fX)<{acceptance_limit[0]}&abs(fY)<{acceptance_limit[1]}").copy()
    lData['fX'] = lData['fX'] * np.sign(lData['fPdgCode'])
    lData['sx'] = np.sqrt((400. - (lData['fX'] + 200.)) / 400.)
    return FCN(weight, lData['sx'].sum(), 0)


def sx_fcn_v2(df_input, snd_input, weight, weight_sc, momentum_limit=1, acceptance_limit=(200, 300)):
    good_t_flux = 2.
    good_snd_flux = 5.

    snd_impact = 1.
    t_impact = 5.
    sc_w_impact = 100.
    w_impact = 1.

    lData = df_input.query(f"P>{momentum_limit}&abs(fX)<{acceptance_limit[0]}&abs(fY)<{acceptance_limit[1]}").copy()
    lData['fX'] = lData['fX'] * np.sign(lData['fPdgCode'])
    lData['sx'] = np.sqrt((400. - (lData['fX'] + 200.)) / 400.)*np.sqrt((600. - (lData['fY'].abs() + 300.)) / 600.)
    t_flux = lData['sx'].sum()
    snd_flux = len(snd_input.query(f"P>{momentum_limit}").index)

    w_part = sc_w_impact * weight_sc + w_impact * weight
    t_part = t_impact*np.maximum(0, (t_flux - good_t_flux))
    snd_part = snd_impact*np.maximum(0, (snd_flux - good_snd_flux))
    return w_part * (t_part + snd_part + 1.)

def get_params_from_json(jsonFile):
    with open(jsonFile) as result_file:
        result = json.load(result_file)
        params = np.array(result['params'])
        params = [float(p) for p in params]
        W = float(result['w'])
        W_sc = float(result['w_sc'])
        return params, W, W_sc


def calc_FCNs(dirName, fileName, nFiels, FCNs_l, weight, sc_weight, tfilter=True, acceptance_limit=(200, 300)):
    FCNs = deepcopy(FCNs_l)
    with_snd = True if 'v2' in FCNs.keys() else False
    basePath = sorted(Path(dirName).glob(f'*/{fileName}'))
    
    assert len(basePath)%nFiels == 0

    branches_to_load_base = ['fluxDetPoint']
    snd_planes = 'TTPoint'
    if with_snd:
        branches_to_load_base.append(snd_planes)
    branches_to_load = {}
    if tfilter:
        branches_to_load_base.append('strawtubesPoint')
    if 'veto' in FCNs.keys():
        branches_to_load_base.append('vetoPoint')
    branches_to_load_base = {i: [k for k in ['fTrackID', 'fPdgCode', 'fX', 'fY', 'fZ', 'fDetectorID', 'fPx', 'fPy', 'fPz', 'fLength']]
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
            events_white_list[bName] =pd.unique(bFlat['event_id'])

        filter_mask = np.isin(events_white_list['T1'], events_white_list['T4'])
        white_list = events_white_list['T1'][filter_mask]
        t2t3_events = np.unique(np.concatenate(
            (events_white_list['T1'], events_white_list['T2']), axis=0))
        filter_mask_23 = np.isin(white_list, t2t3_events)
        white_list = white_list[filter_mask_23]

    vflat = None
    if 'veto' in FCNs.keys():
        vData = mu_only_data['vetoPoint']
        vflat = pd.DataFrame({key: ak.flatten(vData[key], axis=None) for key in vData.keys()})
        vflat.rename(columns={col: col.split('.')[-1] for col in vflat.columns}, inplace=True)
        vflat['P'] = np.sqrt(np.square(vflat['fPx']) + np.square(vflat['fPy']) + np.square(vflat['fPz']))

    data = mu_only_data['fluxDetPoint']
    flat = {key: ak.flatten(data[key], axis=None) for key in data.keys()}
    dfFlat = pd.DataFrame(flat)
    reduced = dfFlat.groupby(['event_id', '{}.fTrackID'.format('fluxDetPoint')], as_index=False).mean()
    reduced.rename(columns={col: col.split('.')[-1] for col in reduced.columns}, inplace=True)
    reduced['P'] = np.sqrt(np.square(reduced['fPx'])+np.square(reduced['fPy'])+np.square(reduced['fPz']))
    reduced['Pt'] = np.sqrt(np.square(reduced['fPx']) + np.square(reduced['fPy']))
    reduced.drop(['fDetectorID', 'fPx', 'fPy', 'fPz'], axis=1, inplace=True)
    if with_snd:
        data_snd = mu_only_data[snd_planes]
        flat_snd = {key: ak.flatten(data_snd[key], axis=None) for key in data_snd.keys()}
        dfFlat_snd = pd.DataFrame(flat_snd)
        reduced_snd = dfFlat_snd.groupby(['event_id', '{}.fTrackID'.format(snd_planes)], as_index=False).mean()
        reduced_snd.rename(columns={col: col.split('.')[-1] for col in reduced_snd.columns}, inplace=True)
        reduced_snd['P'] = np.sqrt(np.square(reduced_snd['fPx']) + np.square(reduced_snd['fPy']) + np.square(reduced_snd['fPz']))
        reduced_snd['Pt'] = np.sqrt(np.square(reduced_snd['fPx']) + np.square(reduced_snd['fPy']))
        reduced_snd.drop(['fDetectorID', 'fPx', 'fPy', 'fPz'], axis=1, inplace=True)

    nRuns = int(len(basePath) / batch_split)
    if with_snd:
        v2_fcn = FCNs.pop('v2')
    outs = {fcn_name: fcn(reduced, weight+sc_weight, acceptance_limit=acceptance_limit) for fcn_name, fcn in FCNs.items() if 'veto' not in fcn_name}
    if with_snd:
        outs['v2'] = v2_fcn(reduced, reduced_snd, weight, sc_weight, momentum_limit=1, acceptance_limit=(200, 300))
    if 'veto' in FCNs.keys():
        outs['veto'] = FCNs['veto'](vflat, weight)
    outs['nRuns'] = nRuns

    #if white_list is not None:
    if tfilter:
        # if white_list is not None:
            reduced = reduced[reduced['event_id'].isin(white_list)]
            for fcn_name, fcn in FCNs.items():
                if 'veto' not in fcn_name and 'sx' in fcn_name or 'count' in fcn_name or 'tCount' in fcn_name:
                    outs[f'{fcn_name}_tracks'] = fcn(reduced, weight+sc_weight, acceptance_limit=acceptance_limit)
            if with_snd:
                outs['v2_tracks'] = v2_fcn(reduced, reduced_snd, weight, sc_weight, momentum_limit=1, acceptance_limit=(200, 300))
        # else:
        #     for fcn_name, fcn in FCNs.items():
        #         if 'veto' not in fcn_name and 'sx' in fcn_name or 'count' in fcn_name:
        #             outs[f'{fcn_name}_tracks'] = 0.
        #     if with_snd:
        #         outs['v2_tracks'] = v2_fcn(reduced, reduced_snd, weight, sc_weight, momentum_limit=1, acceptance_limit=(200, 300))
    return outs


def add_data_from_dir(dirName, FCNs, fileName="ship.conical.MuonBack-TGeant4.root"):
    myRedis = redis.Redis(
        host='localhost',
        port='6379',
        db=db['opt_db'])
    for i, subDir in enumerate(os.listdir(dirName)):
        print(f"Working on {i+1} out of {len(os.listdir(dirName))}")
        try:
            params, W = get_params_from_json(os.path.join(dirName, subDir, "0", "optimise_input.json"))
        except FileNotFoundError:
            continue
        fcn_vals = calc_FCNs(os.path.join(dirName, subDir), fileName, batch_split, FCNs, weight=W)
        print(fcn_vals)
        for fcn_name, fcn_val in fcn_vals.items():
            myRedis.hset(json.dumps(params, cls=NpEncoder), fcn_name, fcn_val)
        myRedis.hset(json.dumps(params, cls=NpEncoder), 'W', W)
        myRedis.hset(json.dumps(params, cls=NpEncoder), 'iteration', subDir)
        myRedis.hset(json.dumps(params), 'tag', "1")

def add_data_from_dir_json(dirName, FCNs, trF, fileName="ship.conical.MuonBack-TGeant4.root", checkStats=False):

    cache = []
    for i, subDir in enumerate(os.listdir(dirName)):
        print(f"Working on {i+1} out of {len(os.listdir(dirName))}")
        try:
            params, W, W_sc = get_params_from_json(os.path.join(dirName, subDir, "0", "optimise_input.json"))
        except FileNotFoundError:
            print("file not found")
            continue
        fcn_vals = calc_FCNs(os.path.join(dirName, subDir), fileName, batch_split, FCNs, weight=W, sc_weight=W_sc, tfilter=trF)
        new_cache_element = {'iteration': i,
                             'W': W,
                             'W_sc': W_sc,
                             'fcns': fcn_vals,
                             'parameters': params,
                             'tag': 1}
        # if checkStats:
        #     new_cache_element.update({'nRuns': job['nRuns']})
        cache.append(new_cache_element)
        if not i % 10:
            with open('new_db.json', 'w') as out:
                json.dump(cache, out)
    with open('new_db.json', 'w') as out:
        json.dump(cache, out)

#myFCNs = {'sx': sx_fcn, 'count': count_fcn}
FCNs = {}
FCNs['tCount'] = tCount_fcn
FCNs['sx'] = sx_fcn
FCNs['v2'] = sx_fcn_v2
FCNs['count'] = count_fcn
FCNs['veto'] = veto_fcn

if __name__ == "__main__":
    add_data_from_dir_json(HOST_LOCALOUTPUT_DIRECTORY, FCNs, trF=add_tracks_filter, checkStats=False)
