from typing import List, Tuple, Any, Dict, Optional, Sequence, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
import os, sys, time
import paramiko
import yaml

class TNMRGradEnv(gym.Env):
    def __init__(self, id:str, action_dim: int = 1,
                  vector_size: Tuple[int] = (130,),
                  dict_obs_space: bool = True,):
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,)) # bounded to reasonable values based on the achievable slew
        self.window_size = vector_size[0]

        self.ideal_waveform = np.squeeze(sio.loadmat('ideal_gradient_pulse.mat')['ideal_p'])
        self.ideal_waveform = np.array(self.ideal_waveform.astype('float'))
        self.ideal_waveform_padded = np.concatenate([np.zeros(vector_size),self.ideal_waveform, np.zeros(vector_size)])
        print(np.shape(self.ideal_waveform_padded))

        self._dict_obs_space = dict_obs_space
        if self._dict_obs_space:
            self.observation_space = gym.spaces.Dict(
                {
                    "pulse": gym.spaces.Box(-100, 100, shape=vector_size, dtype=np.float32),
                    "time": gym.spaces.Box(0.0, np.size(self.ideal_waveform), (1,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(-100, 100, shape=vector_size, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)

        self.preemphasized_waveform = self.ideal_waveform.astype('float')
        self._n_steps = self.ideal_waveform.size
        self._current_step = 0
        self._n_grad_measurement_averages = 1.0
        self.action_scale = 20

        # info for communicating to the TNMR magnet
        self.remote_ip = '10.115.11.91'
        self.remote_username = 'grissomlfi'
        self.remote_password = os.environ['GRISSOM_LFI_PWD']
        self.render_mode='human'

        # constraints
        self.upper_amp_limit = 100
        self.lower_amp_limit = -100

        # set scanner state to false!
        self.scanner_config_dir = os.getcwd() + "tnmr_scanner_state/tnmr_scanner.yaml"
        self.set_scanner_is_occupied(False)



    def step(self, action):
        print(f'CURRENT STEP ={self._current_step}, ACTION = {action}')

        # get the preemphasis. Make sure to clip so that the net waveform does not violate amplitude constraints
 
        #current_ideal_waveform_sample = self.ideal_waveform[:,self._current_step]
        
        action = action * self.action_scale
        self.preemphasis_v[self._current_step] = action

        self.preemphasized_waveform = self.ideal_waveform + self.preemphasis_v

        self.preemphasized_waveform = np.clip(self.preemphasized_waveform, self.lower_amp_limit, self.upper_amp_limit)

        # When you reach the end of the pulse, measure the full waveform and record all relevant information to compute episode reward
        if self._current_step == self._n_steps-1:

            # check yaml to make sure scanner is not currently scanning
            tnmr_occupied = self.get_scanner_is_occupied()

            while(tnmr_occupied):
                time.sleep(2)
                tnmr_occupied = self.get_scanner_is_occupied()
            
            # once free, set as occupied!
            self.set_scanner_is_occupied(is_occupied=True)

            designed_waveform_filename = 'designed_gradient_pulse.mat'
            output_filename = 'current_measurement_data.mat'
            sio.savemat(designed_waveform_filename,{'designed_p':self.preemphasized_waveform})

            print('Measuring on TNMR...')
            try:
                os.remove('current_measurement_data.mat') # remove the old copy of the data
            except OSError:
                pass
            
            self._execute_remote_measurement(designed_waveform_filename, output_filename)

            #matlab_done = self.matlab_eng.iterative_measurement_function(self._n_grad_measurement_averages)
            recorded_data = sio.loadmat(output_filename)
            error_v = recorded_data['error']
            measured_waveform = recorded_data['measured_waveform']
 
            print('Done measuring on TNMR!')
            self.set_scanner_is_occupied(is_occupied=False)

            reward = - np.sum(np.abs(error_v**2))
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
        self._put_file_on_remote(designed_waveform_filename, 'D:/Jonathan/gradient_RL_lowfield/io/'+designed_waveform_filename, verbose=True)

        # Step 2: (On the remote....) Perform the measurement. The design file will be taken up, and a measurement file will be generated.
        # During this time, just wait.
        time.sleep(30)

        # Step 3: Get the measurement files
        self._get_file_from_remote('D:/Jonathan/gradient_RL_lowfield/io/'+output_filename, output_filename, verbose=True)
        return 
        

    def _get_file_from_remote(self, remotepath, filepath, verbose=False):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.remote_ip, username=self.remote_username, password=self.remote_password)
        
        if verbose:
            print(f'Getting file: {filepath}')
        try:
            sftp = client.open_sftp()
            sftp.get(remotepath=remotepath, localpath=filepath)
            sftp.close()

        except:
            print('Getting file from remote failed')
        
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
        if self._dict_obs_space:
            return {
                "pulse": np.squeeze(np.array(ideal_window, dtype=np.float32)),
                "time": np.squeeze(np.array(self._current_step, dtype=np.float)),
            }
        else:
            return np.squeeze(np.array(ideal_window, dtype=np.float32))
            # TODO: must somehow incorporate time into non-dictionary return!

    

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"pulse": obs}
    
    def close(self):
        pass

    def seed(self, seed=None):
        pass
