import os
import json
from kub_config import *
from opt_utils import *#get_jobs_list_client


run_dir = os.path.join(pool_dir, 'running')
new_dir = os.path.join(pool_dir, 'new')
completed_dir = os.path.join(pool_dir, 'completed')

def retrieve_result(uuid):

    #check in running and new
    job_list = get_jobs_list_client(run_dir) + get_jobs_list_client(new_dir)
    job_list_uuid = [job['trial_index'] for job in job_list]
    out = {}
    if uuid in job_list_uuid:
        out['container_status'] = 'running'
        return out
    my_job = [job for job in get_jobs_list_client(run_dir) if job['trial_index'] == uuid]
    if len(my_job) != 1:
        print("Too many jobs or job is not found")
        return out
    else:
        my_job = my_job[0]
        out['container_status'] = 'exited'
        out['veto_points'] = my_job['veto']#None#np.random.rand(nevents, 2)*300.
        out['params'] = params2opt(my_job['parameters'])#None
        out['kinematics'] =  my_job['kinematics'] #None

        # tmp_entry = [13, np.random.randn(), np.random.randn(), -1000., np.random.randn() * 10, np.random.randn() * 10,
        #              np.random.randn() * 200]
        # print(tmp_entry)
        # kinematics.append(tmp_entry)
        #
        # 'kinematics': np.array(kinematics)

    #out['container_status']  = 'exited'  #'failed', 'running'

    return out#{'container_status': 'exited',  'veto_points': np.random.rand(nevents, 2)*300.,  'params': tmp_params, 'kinematics': np.array(kinematics)}#'params': tmp_params,'muons_momentum':np.ones((10, 2)),



def retrieve_params():
    pass#Not used for now

def add_job(job, LdirName):
    print(os.path.join(LdirName, (str(job['trial_index']) + ".json")))
    with open(os.path.join(LdirName, (str(job['trial_index']) + ".json")), 'w') as out:
        json.dump(job, out)

def simulate(old_dict):
    new_dict = old_dict
    new_dict["trial_index"] = old_dict['uuid']
    new_dict['parameters'] = params2sim(new_dict['shape'])
    print("simulate", new_dict)
    add_job(new_dict, new_dir)
    return new_dict["trial_index"]
