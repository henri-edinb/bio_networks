import numpy as np
import pickle
from skimage.color import rgb2gray
from scipy.io import loadmat
import pandas as pd
import random
from random import randrange
from PIL import Image
import cv2
import os
import csv
import math
from skimage.util import view_as_blocks

#crossbar
class MultipleSpeedCrossingBar:

    @staticmethod
    def default_params():
        return {'screen_size': 12, 'bar_size': 1, 'speed': 1, 'noise': 0.1}
    
    def __init__(self, req_screen_size, req_bar_size, req_max_bar_speed, req_noise_freq):
        self.name = 'crossbar'
        self.x_size = req_screen_size
        self.y_size = req_screen_size
        self.num_channels = 1
        self.current_x_bars_position = list()
        self.current_x_bars_direction = list()
        self.current_x_bars_speed = list()
        self.current_y_bars_position = list()
        self.current_y_bars_direction = list()
        self.current_y_bars_speed = list()
        self.speed_max = req_max_bar_speed
        self.bar_size = req_bar_size
        self.noise_freq = req_noise_freq
        random.seed(420)
        self.set_random_position()
        
    def get_current_frame(self):
        #draw x bars
        for x_bar_idx, x_bar_pos in enumerate(self.current_x_bars_position):
            for i in range(0,self.y_size):
                for j in range(0, self.bar_size):
                    if (x_bar_pos + j) < self.x_size:
                        self.current_frame[i][x_bar_pos + j] = 1
        
        #draw y bar
        for y_bar_idx, y_bar_pos in enumerate(self.current_y_bars_position):
            for i in range(0,self.x_size):
                for j in range(0, self.bar_size):
                    if (y_bar_pos + j) < self.y_size:
                        self.current_frame[y_bar_pos + j][i] = 1
        
        
        return self.current_frame
    
    def get_blind_frame(self):
        return np.zeros((self.y_size, self.x_size, 1))
    
    def get_label_for(self, label_category):
        match label_category:
            case 'position_x':
                label_pos_x = np.zeros(self.y_size)
                for _, x_bar_pos in enumerate(self.current_x_bars_position):
                    label_pos_x[x_bar_pos] = 1.0
                return label_pos_x
            case 'position_y':
                label_pos_y = np.zeros(self.x_size)
                for _, y_bar_pos in enumerate(self.current_y_bars_position):
                    label_pos_y[y_bar_pos] = 1.0
                return label_pos_y
            case 'direction_x':
                label_dir_x = np.zeros(2)
                for _, x_bar_dir in enumerate(self.current_x_bars_direction):
                    label_dir_x[x_bar_dir] = 1.0
                return label_dir_x
            case 'direction_y':
                label_dir_y = np.zeros(2)
                for _, y_bar_dir in enumerate(self.current_y_bars_direction):
                    label_dir_y[y_bar_dir] = 1.0
                return label_dir_y
            case 'speed_x':
                label_speed_x = np.zeros(self.speed_max)
                for _, x_bar_speed in enumerate(self.current_x_bars_speed):
                    label_speed_x[x_bar_speed] = 1.0
                return label_speed_x
            case 'speed_y':
                label_speed_y = np.zeros(self.speed_max)
                for _, y_bar_speed in enumerate(self.current_y_bars_speed):
                    label_speed_y[y_bar_speed] = 1.0
                return label_speed_y
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['position_x'] = np.zeros(self.y_size)
        for _, x_bar_pos in enumerate(self.current_x_bars_position):
            label_obj['position_x'][x_bar_pos] = 1.0
        label_obj['position_y'] = np.zeros(self.x_size)
        for _, y_bar_pos in enumerate(self.current_y_bars_position):
            label_obj['position_y'][y_bar_pos] = 1.0
        label_obj['direction_x'] = np.zeros(2)
        for _, x_bar_dir in enumerate(self.current_x_bars_direction):
            label_obj['direction_x'][x_bar_dir] = 1.0
        label_obj['direction_y'] = np.zeros(2)
        for _, y_bar_dir in enumerate(self.current_y_bars_direction):
            label_obj['direction_y'][y_bar_dir] = 1.0
        label_obj['speed_x'] = np.zeros(self.speed_max)
        for _, x_bar_speed in enumerate(self.current_x_bars_speed):
            label_obj['speed_x'][x_bar_speed-1] = 1.0
        label_obj['speed_y'] = np.zeros(self.speed_max)
        for _, y_bar_speed in enumerate(self.current_y_bars_speed):
            label_obj['speed_y'][y_bar_speed-1] = 1.0
        return label_obj
        
    def get_label_list(self):
        return ["position_x", "position_y", "direction_x", "direction_y", "speed_x", "speed_y"]
    
    def get_shape(self):
        return (self.y_size, self.x_size, self.num_channels)

    def set_sampling_distribution(self, new_sd):
        pass

    def set_blind_frame(self):
        a = np.random.rand(self.y_size, self.x_size)
        self.current_frame = a*(a < self.noise_freq)


    def set_random_position(self):
        self.current_x_bars_position = list()
        self.current_x_bars_direction = list()
        self.current_y_bars_position = list()
        self.current_y_bars_direction = list()

        x_speed = randrange(self.speed_max)+1
        x_direction = randrange(2)
        self.current_x_bars_position.append(randrange(self.x_size))
        self.current_x_bars_direction.append(x_direction)
        self.current_x_bars_speed.append(x_speed)

        y_speed = randrange(self.speed_max)+1
        y_direction = randrange(2)
        self.current_y_bars_position.append(randrange(self.y_size))
        self.current_y_bars_direction.append(y_direction)
        self.current_y_bars_speed.append(y_speed)

        a = np.random.rand(self.y_size, self.x_size)
        self.current_frame = a*(a < self.noise_freq)
    
    def set_next_position(self):
        self.change_x()
        self.change_y()

        a = np.random.rand(self.y_size, self.x_size)
        self.current_frame = a*(a < self.noise_freq)
    
    def change_x(self):
        new_x_positions = list()
        for x_bar_idx, x_bar_pos in enumerate(self.current_x_bars_position):
            new_x_pos = 0
            if self.current_x_bars_direction[x_bar_idx] == 0:
                new_x_pos = (x_bar_pos + self.current_x_bars_speed[x_bar_idx]) % self.y_size
            else:
                new_x_pos = (x_bar_pos - self.current_x_bars_speed[x_bar_idx]) % self.y_size
            new_x_positions.append(new_x_pos)
        self.current_x_bars_position = new_x_positions
    
    def change_y(self):
        new_y_positions = list()
        for y_bar_idx, y_bar_pos in enumerate(self.current_y_bars_position):
            new_y_pos = 0
            if self.current_y_bars_direction[y_bar_idx] == 0:
                new_y_pos = (y_bar_pos + self.current_y_bars_speed[y_bar_idx]) % self.x_size
            else:
                new_y_pos = (y_bar_pos - self.current_y_bars_speed[y_bar_idx]) % self.x_size
            new_y_positions.append(new_y_pos)
        self.current_y_bars_position = new_y_positions


#dots
class MovingDots:

    @staticmethod
    def default_params():
        return {'screen_size': 12, 'num_objects': 3, 'noise': 0.1, 'speed': 1}
    
    def __init__(self, req_screen_size, req_number_of_dots, req_noise_freq, req_speed):
        self.name = 'dots'
        
        self.x_size = req_screen_size
        self.y_size = req_screen_size
        self.num_channels = 1
        
        
        self.x_positions = list()
        self.y_positions = list()
        self.x_directions = list()
        self.y_directions = list()
        self.number_of_dots = req_number_of_dots

        for i in range(0, self.number_of_dots):
            self.x_positions.append(np.random.randint(0, self.x_size))
            self.y_positions.append(np.random.randint(0, self.y_size))
            self.x_directions.append(np.random.randint(-req_speed, req_speed))
            self.y_directions.append(np.random.randint(-req_speed, req_speed))
            
        self.max_speed = req_speed
        self.noise_freq = req_noise_freq
        
    def get_current_frame(self):
        for i in range(0, self.number_of_dots):
            self.frame[(self.x_positions[i]+1)%self.x_size][(self.y_positions[i]+1)%self.y_size] = 1
            self.frame[(self.x_positions[i]+1)%self.x_size][(self.y_positions[i])%self.y_size] = 1
            self.frame[(self.x_positions[i]+1)%self.x_size][(self.y_positions[i]-1)%self.y_size] = 1
            self.frame[(self.x_positions[i]+1)%self.x_size][(self.y_positions[i]-2)%self.y_size] = 1
            self.frame[(self.x_positions[i])%self.x_size][(self.y_positions[i]+1)%self.y_size] = 1
            self.frame[(self.x_positions[i])%self.x_size][(self.y_positions[i])%self.y_size] = 1
            self.frame[(self.x_positions[i])%self.x_size][(self.y_positions[i]-1)%self.y_size] = 1
            self.frame[(self.x_positions[i])%self.x_size][(self.y_positions[i]-2)%self.y_size] = 1
            self.frame[(self.x_positions[i]-1)%self.x_size][(self.y_positions[i]+1)%self.y_size] = 1
            self.frame[(self.x_positions[i]-1)%self.x_size][(self.y_positions[i])%self.y_size] = 1
            self.frame[(self.x_positions[i]-1)%self.x_size][(self.y_positions[i]-1)%self.y_size] = 1
            self.frame[(self.x_positions[i]-1)%self.x_size][(self.y_positions[i]-2)%self.y_size] = 1
            self.frame[(self.x_positions[i]-2)%self.x_size][(self.y_positions[i]+1)%self.y_size] = 1
            self.frame[(self.x_positions[i]-2)%self.x_size][(self.y_positions[i])%self.y_size] = 1
            self.frame[(self.x_positions[i]-2)%self.x_size][(self.y_positions[i]-1)%self.y_size] = 1
            self.frame[(self.x_positions[i]-2)%self.x_size][(self.y_positions[i]-2)%self.y_size] = 1

            
        return self.frame
    
    def get_blind_frame(self):
        return self.frame

    def get_label_for(self, label_category):
        match label_category:
            case 'direction_full':
                label_dir_xy = np.zeros((self.max_speed*2)**2)
                dir_index = ((self.current_direction_x+self.max_speed) * (self.max_speed*2)) + (self.current_direction_y+self.max_speed)
                label_dir_xy[dir_index] = 1.0
                return label_dir_xy
            case 'direction_x':
                label_dir_x = np.zeros((self.max_speed*2))
                label_dir_x[self.current_direction_x+self.max_speed] = 1.0
                return label_dir_x
            case 'direction_y':
                label_dir_y = np.zeros((self.max_speed*2))
                label_dir_y[self.current_direction_y+self.max_speed] = 1.0
                return label_dir_y
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['direction_full'] = np.zeros((self.max_speed*2)**2)
        dir_index = ((self.current_direction_x+self.max_speed) * (self.max_speed*2)) + (self.current_direction_y+self.max_speed)
        label_obj['direction_full'][dir_index] = 1.0
        label_obj['direction_x'] = np.zeros((self.max_speed*2))
        label_obj['direction_x'][self.current_direction_x+self.max_speed] = 1.0
        label_obj['direction_y'] = np.zeros((self.max_speed*2))
        label_obj['direction_y'][self.current_direction_y+self.max_speed] = 1.0
        return label_obj
        
    def get_label_list(self):
        return ["direction_full", "direction_x", "direction_y"]
    
    def get_shape(self):
        return (self.y_size, self.x_size, self.num_channels)
    
    def set_sampling_distribution(self, new_sd):
        pass

    def set_random_position(self):
        for i in range(0, self.number_of_dots):
            self.x_positions[i] = np.random.randint(0, self.x_size)
            self.y_positions[i] = np.random.randint(0, self.y_size)
            self.x_directions[i] = np.random.randint(-self.max_speed, self.max_speed)
            self.y_directions[i] = np.random.randint(-self.max_speed, self.max_speed)
        a = np.random.rand(self.y_size, self.x_size)
        self.frame = a*(a < self.noise_freq)
    
    def set_next_position(self):
        for i in range(0, self.number_of_dots):
            new_x = self.x_positions[i] + self.x_directions[i]
            new_y = self.y_positions[i] + self.y_directions[i]
            self.x_positions[i] = new_x % self.x_size
            self.y_positions[i] = new_y % self.y_size
        a = np.random.rand(self.y_size, self.x_size)
        self.frame = a*(a < self.noise_freq)


#gratings
class DriftingGratings:
    @staticmethod
    def default_params():
        return {
            'screen_size': 32,
            'spatial_period': 8.0,
            'max_phase_speed': 2,
            'num_directions': 8,
            'contrast': 1.0,
            'mean_luminance': 0.5,
            'square_wave': False,
            'noise_freq': 0.0,
        }

    def __init__(self, hparams):
        self.name = 'gratings'
        self.x_size = hparams['screen_size']
        self.y_size = hparams['screen_size']
        self.num_channels = 1

        self.spatial_period = float(hparams.get('spatial_period', 8.0))
        self.speed_max = int(hparams.get('max_phase_speed', 2))
        self.num_directions = int(hparams.get('num_directions', 8))
        self.contrast = float(hparams.get('contrast', 1.0))
        self.mean_luminance = float(hparams.get('mean_luminance', 0.5))
        self.square_wave = bool(hparams.get('square_wave', False))
        self.noise_freq = float(hparams.get('noise_freq', 0.0))

        if self.spatial_period <= 0:
            raise ValueError('spatial_period must be > 0')
        if self.speed_max < 1:
            raise ValueError('max_phase_speed must be >= 1')
        if self.num_directions < 2:
            raise ValueError('num_directions must be >= 2')
        if not 0.0 <= self.contrast <= 1.0:
            raise ValueError('contrast must be between 0 and 1')
        if not 0.0 <= self.mean_luminance <= 1.0:
            raise ValueError('mean_luminance must be between 0 and 1')

        self.direction_angles = np.linspace(0.0, 2.0 * np.pi, self.num_directions, endpoint=False)
        self.possible_speeds = list(range(1, self.speed_max + 1))
        self.speed_to_index = {s: i for i, s in enumerate(self.possible_speeds)}

        self.current_phase = 0.0
        self.current_direction_idx = 0
        self.current_speed = 1

        random.seed(420)
        self.set_random_position()

    def _phase_increment(self):
        return (2.0 * np.pi * self.current_speed) / self.spatial_period

    def _phase_bin(self):
        return int(np.floor((self.current_phase % (2.0 * np.pi)) / (2.0 * np.pi) * self.x_size)) % self.x_size

    def set_random_position(self):
        self.current_phase = random.uniform(0.0, 2.0 * np.pi)
        self.current_direction_idx = random.randrange(self.num_directions)
        self.current_speed = random.choice(self.possible_speeds)

    def set_test_position(self):
        self.set_random_position()

    def set_validation_position(self):
        self.set_random_position()

    def set_next_position(self):
        self.current_phase = (self.current_phase + self._phase_increment()) % (2.0 * np.pi)

    def get_current_frame(self):
        yy, xx = np.indices((self.y_size, self.x_size), dtype=float)
        angle = self.direction_angles[self.current_direction_idx]
        nx = np.cos(angle)
        ny = np.sin(angle)

        projected = xx * nx + yy * ny
        phase_map = (2.0 * np.pi * projected / self.spatial_period) - self.current_phase

        if self.square_wave:
            wave = np.sign(np.sin(phase_map))
            wave[wave == 0] = 1.0
        else:
            wave = np.sin(phase_map)

        frame = self.mean_luminance + 0.5 * self.contrast * wave
        frame = np.clip(frame, 0.0, 1.0)

        if self.noise_freq > 0.0:
            a = np.random.rand(self.y_size, self.x_size)
            frame = np.clip(frame + a * (a < self.noise_freq), 0.0, 1.0)

        return frame

    def get_blind_frame(self):
        return np.zeros((self.y_size, self.x_size))

    def get_label_for(self, label_category):
        phase_bin = self._phase_bin()
        angle_deg = (360.0 * self.current_direction_idx / self.num_directions) % 360.0

        match label_category:
            case 'category_string':
                return f'direction{self.current_direction_idx}'
            case 'category_one_hot':
                label = np.zeros(self.num_directions)
                label[self.current_direction_idx] = 1.0
                return label
            case 'phase':
                label = np.zeros(self.x_size)
                label[phase_bin] = 1.0
                return label
            case 'direction':
                label = np.zeros(self.num_directions)
                label[self.current_direction_idx] = 1.0
                return label
            case 'speed':
                label = np.zeros(len(self.possible_speeds))
                label[self.speed_to_index[self.current_speed]] = 1.0
                return label
            case 'direction_angle_deg':
                return np.array([angle_deg], dtype=float)
            case 'phase_rad':
                return np.array([self.current_phase], dtype=float)
            case _:
                return None

    def get_all_labels(self):
        return {
            'category_string': self.get_label_for('category_string'),
            'category_one_hot': self.get_label_for('category_one_hot'),
            'phase': self.get_label_for('phase'),
            'direction': self.get_label_for('direction'),
            'speed': self.get_label_for('speed'),
            'direction_angle_deg': self.get_label_for('direction_angle_deg'),
            'phase_rad': self.get_label_for('phase_rad'),
        }

    def get_label_list(self):
        return [
            'category_string', 'category_one_hot', 'phase',
            'direction', 'speed', 'direction_angle_deg', 'phase_rad'
        ]

    def get_action_label(self):
        return np.concatenate((self.get_label_for('direction'), self.get_label_for('speed')))

    def get_hyperparameters(self):
        return {
            'vg_name': self.name,
            'screen_size': self.x_size,
            'spatial_period': self.spatial_period,
            'speed_max': self.speed_max,
            'num_directions': self.num_directions,
            'contrast': self.contrast,
            'mean_luminance': self.mean_luminance,
            'square_wave': self.square_wave,
            'noise': self.noise_freq,
        }

    def get_extensive_name(self):
        return (
            f'{self.name}_{self.spatial_period}_{self.speed_max}_'
            f'{self.num_directions}_{self.contrast}_{self.square_wave}_{self.noise_freq}'
        )

    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)

    def get_category_string_from_one_hot(self, one_hot):
        return f'direction{int(np.argmax(one_hot))}'

    def get_name(self):
        return self.name

#dot_motion
class DotMotion:
    @staticmethod
    def default_params():
        return {
            'screen_size': 32,
            'n_dots': 64,
            'dot_radius': 1,
            'max_dot_speed': 2,
            'num_directions': 16,
            'coherence': 1.0,
            'noise_freq': 0.0,
        }

    def __init__(self, hparams):
        self.name = 'dot_motion'
        self.x_size = int(hparams['screen_size'])
        self.y_size = int(hparams['screen_size'])
        self.num_channels = 1

        self.n_dots = int(hparams['n_dots'])
        self.dot_radius = int(hparams['dot_radius'])
        self.speed_max = int(hparams['max_dot_speed'])
        self.num_directions = int(hparams.get('num_directions', 16))
        self.coherence = float(hparams.get('coherence', 1.0))
        self.noise_freq = float(hparams.get('noise_freq', 0.0))

        if self.n_dots <= 0:
            raise ValueError('n_dots must be > 0')
        if self.dot_radius < 0:
            raise ValueError('dot_radius must be >= 0')
        if self.speed_max < 1:
            raise ValueError('max_dot_speed must be >= 1')
        if self.num_directions < 1:
            raise ValueError('num_directions must be >= 1')
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError('coherence must be between 0 and 1')

        self.possible_speeds = list(range(1, self.speed_max + 1))
        self.possible_directions = [
            2.0 * np.pi * idx / self.num_directions for idx in range(self.num_directions)
        ]
        self.direction_to_index = {
            round(float(angle), 12): idx for idx, angle in enumerate(self.possible_directions)
        }
        self.speed_to_index = {speed: idx for idx, speed in enumerate(self.possible_speeds)}

        self.current_direction_idx = 0
        self.current_direction = self.possible_directions[self.current_direction_idx]
        self.current_speed = self.possible_speeds[0]
        self.dot_positions = np.zeros((self.n_dots, 2), dtype=float)

        random.seed(420)
        self.set_random_position()

    def _direction_key(self, angle):
        return round(float(angle), 12)

    def _velocity_from(self, direction, speed):
        return speed * np.cos(direction), speed * np.sin(direction)

    def _randomize_dots(self):
        self.dot_positions[:, 0] = np.random.uniform(0, self.x_size, size=self.n_dots)
        self.dot_positions[:, 1] = np.random.uniform(0, self.y_size, size=self.n_dots)

    def _centroid_bin_x(self):
        return int(np.floor(np.mean(self.dot_positions[:, 0]))) % self.x_size

    def _centroid_bin_y(self):
        return int(np.floor(np.mean(self.dot_positions[:, 1]))) % self.y_size

    def set_random_position(self):
        self._randomize_dots()
        self.current_direction_idx = random.randrange(self.num_directions)
        self.current_direction = self.possible_directions[self.current_direction_idx]
        self.current_speed = random.choice(self.possible_speeds)

    def set_test_position(self):
        self.set_random_position()

    def set_validation_position(self):
        self.set_random_position()

    def set_next_position(self):
        coherent_mask = np.random.rand(self.n_dots) < self.coherence

        vx, vy = self._velocity_from(self.current_direction, self.current_speed)
        self.dot_positions[coherent_mask, 0] += vx
        self.dot_positions[coherent_mask, 1] += vy

        incoherent_idx = np.where(~coherent_mask)[0]
        n_incoherent = int(incoherent_idx.size)
        if n_incoherent > 0:
            random_dirs = np.random.randint(0, self.num_directions, size=n_incoherent)
            random_speeds = np.random.randint(1, self.speed_max + 1, size=n_incoherent)
            random_angles = np.asarray(self.possible_directions, dtype=float)[random_dirs]
            self.dot_positions[incoherent_idx, 0] += random_speeds * np.cos(random_angles)
            self.dot_positions[incoherent_idx, 1] += random_speeds * np.sin(random_angles)

        self.dot_positions[:, 0] %= self.x_size
        self.dot_positions[:, 1] %= self.y_size

    def get_current_frame(self):
        a = np.random.rand(self.y_size, self.x_size)
        frame = a * (a < self.noise_freq)

        for dot_x, dot_y in self.dot_positions:
            cx = int(np.floor(dot_x))
            cy = int(np.floor(dot_y))
            for dy in range(-self.dot_radius, self.dot_radius + 1):
                for dx in range(-self.dot_radius, self.dot_radius + 1):
                    if dx * dx + dy * dy <= self.dot_radius * self.dot_radius:
                        frame[(cy + dy) % self.y_size, (cx + dx) % self.x_size] = 1.0

        return frame

    def get_blind_frame(self):
        return np.zeros((self.y_size, self.x_size), dtype=float)

    def get_label_for(self, label_category):
        direction_index = self.current_direction_idx
        speed_index = self.speed_to_index[self.current_speed]
        centroid_x = self._centroid_bin_x()
        centroid_y = self._centroid_bin_y()

        match label_category:
            case 'category_string':
                return f'{self.current_direction_idx}_dir'
            case 'category_one_hot':
                label = np.zeros(self.num_directions)
                label[direction_index] = 1.0
                return label
            case 'direction':
                label = np.zeros(self.num_directions)
                label[direction_index] = 1.0
                return label
            case 'speed':
                label = np.zeros(len(self.possible_speeds))
                label[speed_index] = 1.0
                return label
            case 'position_x':
                label = np.zeros(self.x_size)
                label[centroid_x] = 1.0
                return label
            case 'position_y':
                label = np.zeros(self.y_size)
                label[centroid_y] = 1.0
                return label
            case 'coherence':
                return np.array([self.coherence], dtype=float)
            case 'direction_angle_deg':
                return np.array([(np.degrees(self.current_direction) % 360.0)], dtype=float)
            case 'direction_x':
                return np.array([np.cos(self.current_direction)], dtype=float)
            case 'direction_y':
                return np.array([np.sin(self.current_direction)], dtype=float)
            case _:
                return None

    def get_all_labels(self):
        return {
            'category_string': self.get_label_for('category_string'),
            'category_one_hot': self.get_label_for('category_one_hot'),
            'direction': self.get_label_for('direction'),
            'speed': self.get_label_for('speed'),
            'position_x': self.get_label_for('position_x'),
            'position_y': self.get_label_for('position_y'),
            'coherence': self.get_label_for('coherence'),
            'direction_angle_deg': self.get_label_for('direction_angle_deg'),
        }

    def get_label_list(self):
        return ['category_string', 'category_one_hot', 'direction', 'speed', 'position_x', 'position_y', 'coherence', 'direction_angle_deg']

    def get_action_label(self):
        return np.concatenate((self.get_label_for('direction'), self.get_label_for('speed')))

    def get_hyperparameters(self):
        return {
            'vg_name': self.name,
            'screen_size': self.x_size,
            'n_dots': self.n_dots,
            'dot_radius': self.dot_radius,
            'max_dot_speed': self.speed_max,
            'num_directions': self.num_directions,
            'coherence': self.coherence,
            'noise': self.noise_freq,
        }

    def get_extensive_name(self):
        return (
            f'{self.name}_{self.n_dots}_{self.dot_radius}_'
            f'{self.speed_max}_{self.num_directions}_{self.coherence}_{self.noise_freq}'
        )

    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)

    def get_category_string_from_one_hot(self, one_hot):
        return f'{int(np.argmax(one_hot))}_dir'

    def get_name(self):
        return self.name



class DriftingSingleBar:
    @staticmethod
    def default_params():
        return {
            'screen_size': 32,
            'bar_width': 5.0,
            'bar_length': None,
            'max_phase_speed': 2,
            'num_directions': 16,
            'contrast': 1.0,
            'mean_luminance': 0.0,
            'soft_edges': True,
            'noise_freq': 0.0,
            'wrap_gap': 32,
        }

    def __init__(self, hparams):
        self.name = 'drifting_single_bar'
        self.x_size = int(hparams['screen_size'])
        self.y_size = int(hparams['screen_size'])
        self.num_channels = 1

        self.bar_width = float(hparams.get('bar_width', 5.0))
        self.bar_length = hparams.get('bar_length', None)
        self.speed_max = int(hparams.get('max_phase_speed', 2))
        self.num_directions = int(hparams.get('num_directions', 16))
        self.contrast = float(hparams.get('contrast', 1.0))
        self.mean_luminance = float(hparams.get('mean_luminance', 0.0))
        self.soft_edges = bool(hparams.get('soft_edges', True))
        self.noise_freq = float(hparams.get('noise_freq', 0.0))
        self.wrap_gap = hparams.get('wrap_gap', None)

        if self.speed_max < 1:
            raise ValueError('max_phase_speed must be >= 1')
        if self.num_directions < 1:
            raise ValueError('num_directions must be >= 1')
        if self.bar_width <= 0:
            raise ValueError('bar_width must be > 0')
        if self.bar_length is not None and float(self.bar_length) <= 0:
            raise ValueError('bar_length must be > 0 or None')
        if self.wrap_gap is not None and float(self.wrap_gap) < 0:
            raise ValueError('wrap_gap must be >= 0 or None')

        self.possible_speeds = list(range(1, self.speed_max + 1))
        self.possible_directions = [
            2.0 * np.pi * idx / self.num_directions for idx in range(self.num_directions)
        ]
        self.direction_to_index = {
            round(float(angle), 12): idx for idx, angle in enumerate(self.possible_directions)
        }
        self.speed_to_index = {speed: idx for idx, speed in enumerate(self.possible_speeds)}

        self.current_center_x = 0.0
        self.current_center_y = 0.0
        self.current_direction = self.possible_directions[0]
        self.current_speed = self.possible_speeds[0]

        yy, xx = np.indices((self.y_size, self.x_size), dtype=float)
        self._xx = xx
        self._yy = yy
        self._screen_diag = float(np.hypot(self.x_size, self.y_size))
        self._tile_offsets = [
            (float(x_off), float(y_off))
            for x_off in (-self.x_size, 0.0, self.x_size)
            for y_off in (-self.y_size, 0.0, self.y_size)
        ]

        random.seed(420)
        self.set_random_position()

    def _direction_key(self, angle):
        return round(float(angle), 12)

    def _velocity_components(self):
        vx = self.current_speed * np.cos(self.current_direction)
        vy = self.current_speed * np.sin(self.current_direction)
        return vx, vy

    def _drift_normal(self):
        return np.cos(self.current_direction), np.sin(self.current_direction)

    def _length_for_rendering(self):
        if self.bar_length is None:
            return None
        return float(self.bar_length)

    def _projected_span_along_normal(self):
        nx, ny = self._drift_normal()
        return abs(nx) * max(self.x_size - 1, 1) + abs(ny) * max(self.y_size - 1, 1)

    def _full_length_period(self):
        gap = self._screen_diag if self.wrap_gap is None else float(self.wrap_gap)
        return self._projected_span_along_normal() + self.bar_width + gap

    def _wrapped_signed_distance(self, values, center, period):
        return (values - center + period / 2.0) % period - period / 2.0

    def _full_length_bar_profile(self):
        nx, ny = self._drift_normal()
        proj = self._xx * nx + self._yy * ny
        center_proj = self.current_center_x * nx + self.current_center_y * ny
        period = self._full_length_period()
        dist_across = self._wrapped_signed_distance(proj, center_proj, period)

        if self.soft_edges:
            sigma_width = max(self.bar_width / 2.0, 1e-6)
            return np.exp(-0.5 * (dist_across / sigma_width) ** 2)

        return (np.abs(dist_across) <= (self.bar_width / 2.0)).astype(float)

    def _finite_bar_profile(self):
        length = self._length_for_rendering()
        tangent_x = -np.sin(self.current_direction)
        tangent_y = np.cos(self.current_direction)

        sample_step = 0.5 if not self.soft_edges else max(0.5, self.bar_width / 4.0)
        n_samples = max(2, int(np.ceil(length / sample_step)) + 1)
        ts = np.linspace(-length / 2.0, length / 2.0, n_samples)

        base_x = self.current_center_x + ts * tangent_x
        base_y = self.current_center_y + ts * tangent_y

        profile = np.zeros((self.y_size, self.x_size), dtype=float)
        radius = self.bar_width / 2.0
        sigma = max(radius, 1e-6)

        for x_off, y_off in self._tile_offsets:
            pts_x = base_x + x_off
            pts_y = base_y + y_off

            dx = self._xx[None, :, :] - pts_x[:, None, None]
            dy = self._yy[None, :, :] - pts_y[:, None, None]
            min_dist2 = np.min(dx * dx + dy * dy, axis=0)

            if self.soft_edges:
                candidate = np.exp(-0.5 * (min_dist2 / (sigma ** 2)))
            else:
                candidate = (min_dist2 <= (radius ** 2)).astype(float)

            profile = np.maximum(profile, candidate)

        return profile

    def _bar_profile(self):
        if self.bar_length is None:
            return self._full_length_bar_profile()
        return self._finite_bar_profile()

    def set_random_position(self):
        self.current_center_x = float(np.random.uniform(0, self.x_size))
        self.current_center_y = float(np.random.uniform(0, self.y_size))
        self.current_direction = random.choice(self.possible_directions)
        self.current_speed = random.choice(self.possible_speeds)

    def set_test_position(self):
        self.set_random_position()

    def set_validation_position(self):
        self.set_random_position()

    def set_next_position(self):
        vx, vy = self._velocity_components()
        self.current_center_x += vx
        self.current_center_y += vy

        if self.bar_length is not None:
            self.current_center_x %= self.x_size
            self.current_center_y %= self.y_size

    def get_current_frame(self):
        if self.noise_freq > 0:
            a = np.random.rand(self.y_size, self.x_size)
            frame = a * (a < self.noise_freq)
        else:
            frame = np.zeros((self.y_size, self.x_size), dtype=float)

        bar = self._bar_profile()
        frame = np.clip(frame + self.mean_luminance + self.contrast * bar, 0.0, 1.0)
        return frame

    def get_blind_frame(self):
        return np.zeros((self.y_size, self.x_size), dtype=float)

    def get_label_for(self, label_category):
        direction_index = self.direction_to_index[self._direction_key(self.current_direction)]
        speed_index = self.speed_to_index[self.current_speed]
        x_bin = int(np.floor(self.current_center_x)) % self.x_size
        y_bin = int(np.floor(self.current_center_y)) % self.y_size

        match label_category:
            case 'category_string':
                angle_deg = (np.degrees(self.current_direction) % 360.0)
                return f'{angle_deg:.1f}deg_speed{self.current_speed}_x{x_bin}_y{y_bin}'
            case 'direction':
                label = np.zeros(self.num_directions)
                label[direction_index] = 1.0
                return label
            case 'speed':
                label = np.zeros(len(self.possible_speeds))
                label[speed_index] = 1.0
                return label
            case 'position_x':
                label = np.zeros(self.x_size)
                label[x_bin] = 1.0
                return label
            case 'position_y':
                label = np.zeros(self.y_size)
                label[y_bin] = 1.0
                return label
            case 'direction_angle_deg':
                return np.array([(np.degrees(self.current_direction) % 360.0)], dtype=float)
            case _:
                return None

    def get_all_labels(self):
        return {
            'category_string': self.get_label_for('category_string'),
            'direction': self.get_label_for('direction'),
            'speed': self.get_label_for('speed'),
            'position_x': self.get_label_for('position_x'),
            'position_y': self.get_label_for('position_y'),
            'direction_angle_deg': self.get_label_for('direction_angle_deg'),
        }

    def get_label_list(self):
        return ['category_string', 'direction', 'speed', 'position_x', 'position_y', 'direction_angle_deg']

    def get_action_label(self):
        return np.concatenate((self.get_label_for('direction'), self.get_label_for('speed')))

    def get_hyperparameters(self):
        return {
            'vg_name': self.name,
            'screen_size': self.x_size,
            'bar_width': self.bar_width,
            'bar_length': self.bar_length,
            'max_phase_speed': self.speed_max,
            'num_directions': self.num_directions,
            'contrast': self.contrast,
            'mean_luminance': self.mean_luminance,
            'soft_edges': self.soft_edges,
            'noise': self.noise_freq,
            'wrap_gap': self.wrap_gap,
        }

    def get_extensive_name(self):
        return (
            f'{self.name}_{self.bar_width}_{self.bar_length}_'
            f'{self.speed_max}_{self.num_directions}_{self.noise_freq}_{self.wrap_gap}'
        )

    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)

    def get_category_string_from_one_hot(self, one_hot):
        return self.get_label_for('category_string')

    def get_name(self):
        return self.name
