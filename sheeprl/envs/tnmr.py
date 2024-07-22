from typing import List, Tuple, Any, Dict, Optional, Sequence, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
import os, sys, time, random
import paramiko
import yaml

class TNMRGradEnv(gym.Env):
    def __init__(self, id:str, action_dim: int = 1,
                  vector_size: Tuple[int] = (130,),
                  dict_obs_space: bool = True,):
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,)) # bounded to reasonable values based on the achievable slew
        self.window_size = vector_size[0]

        self.ideal_waveform = np.squeeze(sio.loadmat('ideal_bipolar.mat')['ideal_p'])
        self.ideal_waveform = np.array(self.ideal_waveform.astype('float'))
        self.ideal_waveform_padded = np.concatenate([np.zeros(vector_size),self.ideal_waveform, np.zeros(vector_size)])

        self.preemphasis_v = np.zeros(np.shape(self.ideal_waveform))
        self.preemphasis_v_padded = np.concatenate([np.zeros(self.window_size),self.preemphasis_v, np.zeros(self.window_size)])

        self._dict_obs_space = dict_obs_space
        if self._dict_obs_space:
            self.observation_space = gym.spaces.Dict(
                {
                    "pulse": gym.spaces.Box(-100, 100, shape=vector_size, dtype=np.float32),
                    "time": gym.spaces.Box(0.0, np.size(self.ideal_waveform), (1,), dtype=np.float32),
                    #"preemph": gym.spaces.Box(-100, 100, shape=vector_size, dtype=np.float32),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(-100, 100, shape=vector_size, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)

        self.preemphasized_waveform = self.ideal_waveform.astype('float')
        self._n_steps = self.ideal_waveform.size
        self._current_step = 0
        self._n_grad_measurement_averages = 1.0
        self.action_scale = 35

        # this determines how often you actually collect data on TNMR
        self.measure_interval = 50
        self.BASE_COST = 10

        # info for communicating to the TNMR magnet
        self.remote_ip = '10.115.11.91'
        self.remote_username = 'grissomlfi'
        self.remote_password = os.environ['GRISSOM_LFI_PWD']
        self.render_mode='None'

        # constraints
        self.upper_amp_limit = 100
        self.lower_amp_limit = -100

        self.env_scanning_id = random.randint(1, 100000)

        self.init_seed = random.randint(1,100000)



    def step(self, action):
        print(f'CURRENT STEP ={self._current_step}, ACTION = {action}')

        # get the preemphasis. Make sure to clip so that the net waveform does not violate amplitude constraints
 
        #current_ideal_waveform_sample = self.ideal_waveform[:,self._current_step]
        
        action = action * self.action_scale
        self.preemphasis_v[self._current_step] = action
        self.preemphasis_v_padded = np.concatenate([np.zeros(self.window_size),self.preemphasis_v, np.zeros(self.window_size)])


        self.preemphasized_waveform = self.ideal_waveform + self.preemphasis_v

        self.preemphasized_waveform = np.clip(self.preemphasized_waveform, self.lower_amp_limit, self.upper_amp_limit)

        # When you reach the end of the pulse, measure the full waveform and record all relevant information to compute episode reward
        if (self._current_step > 0) and (np.mod(self._current_step, self.measure_interval) == 0):
            

            designed_waveform_filename = 'designed_gradient_pulse_'+str(self.env_scanning_id)+'.mat'
            output_filename = 'current_measurement_data_'+str(self.env_scanning_id)+'.mat'
            sio.savemat(designed_waveform_filename,{'designed_p':self.preemphasized_waveform})

            print('Measuring on TNMR...')
            try:
                os.remove(output_filename) # remove the old copy of the data
            except OSError:
                pass
            
            self._execute_remote_measurement(designed_waveform_filename, output_filename)

            recorded_data = sio.loadmat(output_filename)
            error_v = recorded_data['error']
            measured_waveform = recorded_data['measured_waveform']
 
            # reward is the cost up to the end of the window: 
            print('Done measuring on TNMR!')
            #reward_v = - np.abs(error_v[(self._current_step-self.measure_interval):self._current_step]**2)
            #reward = np.sum(reward_v)
            reward = - np.sum(np.abs(error_v**2))/self._n_steps
            print(f'NET ERROR: {np.sum(np.abs(error_v**2))}')
            reward = reward + self.BASE_COST # set baseline so that cost is 0 
            print(f'REWARD ={reward}')
        else: 
            reward = 0

        done = self._current_step == self._n_steps-1
        self._current_step += 1

        observation = self.get_obs()
        #plt.plot(np.transpose(observation))
        #plt.show()

        return (
            observation,
            reward,
            done,
            False,
            {},
        )

    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        seed = self.init_seed
        super().reset(seed=seed)

        # reset to the beginning of the waveform, and set our preemphasis to 0
        self._current_step = 0
        self.preemphasized_waveform = self.ideal_waveform
        self.preemphasis_v = np.zeros(np.shape(self.ideal_waveform))

        observation = self.get_obs()

        return observation, {}
    
    def get_scanner_is_occupied(self):
        # just gets whether the scanner is in use right now or not
        with open(self.scanner_config_dir) as f:
            cfg_tnmr = yaml.load(f, Loader=yaml.FullLoader)

        return cfg_tnmr["scanner_occupied"]

    def set_scanner_is_occupied(self, is_occupied):
        # sets the status of the scanner so that other processes know not to scan

        # load yaml
        with open(self.scanner_config_dir) as f:
            cfg_tnmr = yaml.load(f, Loader=yaml.FullLoader)

        # modify parameter
        cfg_tnmr["scanner_occupied"] = is_occupied
 
        # write yaml
        with open(self.scanner_config_dir, "w") as f:
            cfg_tnmr = yaml.dump(
                cfg_tnmr, stream=f, default_flow_style=False, sort_keys=False
            )

        return
    

    def render(self, mode="human", close=False):
        pass

    def _execute_remote_measurement(self, designed_waveform_filename, output_filename):

        # Step 1: Put the designed waveform file on the remote (TNMR)
        self._put_file_on_remote(designed_waveform_filename, 'D:/Jonathan/gradient_RL_lowfield/io/input/'+designed_waveform_filename, verbose=True)

        # Step 2: (On the remote....) Perform the measurement. The design file will be taken up, and a measurement file will be generated.
        # During this time, just wait. This takes a minimum of 30 seconds so just sleep during that time... 
        time.sleep(25)

        # Step 3: Get the measurement files
        self._get_file_from_remote('D:/Jonathan/gradient_RL_lowfield/io/output/'+output_filename, output_filename, verbose=True)
        return 
        

    def _get_file_from_remote(self, remotepath, filepath, verbose=False):
        # waits until the file is present on the other computer, then gets it here 
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.remote_ip, username=self.remote_username, password=self.remote_password)
        sftp = client.open_sftp()

        file_present = False

        if verbose:
            print(f'Getting file: {filepath}')
        while not file_present:
            time.sleep(10)
            try:
                sftp.get(remotepath=remotepath, localpath=filepath)
                sftp.remove(remotepath)
                sftp.close()

            except IOError:
                print('File not ready on remote. Waiting ... ')

            else: 
                file_present = True
                print('Getting file from remote succeeded! ')
                sftp.close()
        
        client.close()

    def _put_file_on_remote(self, filepath, remotepath, verbose=False):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.remote_ip, username=self.remote_username, password=self.remote_password)
        
        if verbose:
            print(f'Putting file: {filepath}')
        try:
            sftp = client.open_sftp()
            sftp.put(localpath=filepath, remotepath=remotepath)
            sftp.close()

        except:
            print('Putting file to remote failed')
        
        client.close()

    
    def get_obs(self) -> Dict[str, np.ndarray]: 
        current_point_in_padded = self._current_step + self.window_size
        ideal_window = self.ideal_waveform_padded[(current_point_in_padded-self.window_size//2):(current_point_in_padded + self.window_size//2)]
        preemphasis_window = self.preemphasis_v_padded[(current_point_in_padded-self.window_size//2):(current_point_in_padded + self.window_size//2)]
        
        if self._dict_obs_space:
            return {
                "pulse": np.squeeze(np.array(ideal_window, dtype=np.float32)),
                "time": np.squeeze(np.array(self._current_step, dtype=np.float)),
                #"preemph": np.squeeze(np.array(preemphasis_window, dtype=np.float32)),

            }
        else:
            return np.squeeze(np.array(ideal_window, dtype=np.float32))
            # TODO: must somehow incorporate time into non-dictionary return!

    

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"pulse": obs}
    
    def close(self):
        pass

    #def seed(self, seed=None):
    #    pass
