from typing import List, Tuple, Any, Dict, Optional, Sequence, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
import os, sys, time
import paramiko

class TNMRGradEnv(gym.Env):
    def __init__(self, action_dim: int = 1, size: Tuple[int] = (1362,)):
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,)) # bounded to reasonable values based on the achievable slew
        self.observation_space = gym.spaces.Box(-100, 100, shape=size, dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self.ideal_waveform = sio.loadmat('ideal_gradient_pulse.mat')['ideal_p']
        self.preemphasized_waveform = self.ideal_waveform
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

        # initialize matlab
        #eng = matlab.engine.start_matlab()
        #self.matlab_eng = eng
        #print('TNMR GRAD ENVIRONMENT INITIALIZED')



    def step(self, action):
        print(f'CURRENT STEP ={self._current_step}, ACTION = {action}')

        # get the preemphasis. Make sure to clip so that the net waveform does not violate amplitude constraints
 
        current_ideal_waveform_sample = self.ideal_waveform[:,self._current_step]
        
        action = action * self.action_scale
        self.preemphasis_v[:,self._current_step] = action

        self.preemphasized_waveform = self.ideal_waveform + self.preemphasis_v

        self.preemphasized_waveform = np.clip(self.preemphasized_waveform, self.lower_amp_limit, self.upper_amp_limit)

        # When you reach the end of the pulse, measure the full waveform and record all relevant information to compute episode reward
        if self._current_step == self._n_steps-1:

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

            reward = - np.sum(np.abs(error_v**2))
            print(f'REWARD ={reward}')
        else:
            reward = 0

        done = self._current_step == self._n_steps-1
        self._current_step += 1

        observation = self._convert_obs(self._get_obs())
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
        self.preemphasis_v = np.zeros((1,self._n_steps))
        self.preemphasized_waveform = self.ideal_waveform

        observation = self._convert_obs(self._get_obs())

        return observation, {}
    
    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def render(self, mode=None, close=False):
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


    def _run_remote_measurement(self):
        pass
    
    def _get_obs(self):
        #return np.concatenate((self.preemphasis_v, self.ideal_waveform), axis=0)
        return self.preemphasized_waveform
    

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"pulse": obs}
    
    def close(self):
        pass

    def seed(self, seed=None):
        pass
