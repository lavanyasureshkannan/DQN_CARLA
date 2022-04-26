#!/usr/bin/env python

import glob
import math
import os
import sys
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Activation
from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time

IMG_WIDTH = 640
IMG_HEIGHT = 480
SHOW_OUTPUT = False
EPISODE_LENGTH_IN_SECONDS = 12


class VehicleEnv:
    SHOW_RGB_CAM = SHOW_OUTPUT
    img_width = IMG_WIDTH
    img_height = IMG_HEIGHT
    STEER_AMT = 1.0
    front_camera = None

    def __int__(self):
        self.client = carla.Client('localhost', 2000)
        self.timeout = self.client.set_timeout(3.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.car = self.blueprint_library.filter('tt')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.vehicle_spawn_pt = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.car, self.vehicle_spawn_pt)

        self.vehicle_rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.vehicle_rgb_cam.set_attribute("image_size_x", f"{self.img_width}")
        self.vehicle_rgb_cam.set_attribute("image_size_y", f"{self.img_height}")
        self.vehicle_rgb_cam.set_attribute("fov", f"105")

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # RGB Camera sensor initialization
        self.rgb_camera = self.world.spawn_actor(self.vehicle_rgb_cam, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda rgb_data: self.pre_process_img(rgb_data))

        self.vehicle.apply_control(throttle=0.0, brake=0.0)
        time.sleep(3.5)

        # lane_invade = self.blueprint_library.find("sensor.other.")

        # Collision sensor initialization
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_event(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(throttle=0.0, brake=0.0)

        return self.front_camera

    def collision_event(self, event):
        self.collision_hist.append(event)

    def pre_process_img(self, img):
        img1 = np.array(img.raw_data)
        # print(img1.shape)
        im_rgba = img1.reshape((self.img_height, self.img_width, 4))
        im_rgb = im_rgba[:, :, :3]
        # print(im_rgb.shape)
        if self.SHOW_RGB_CAM:
            cv2.imshow("RGB Sensor Output", im_rgb)
            cv2.waitKey(1)
        self.front_camera = im_rgb/255.0
        # return im_rgb/255.0  # Returning values between 0 nd 1

    def action_step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=1 * self.STEER_AMT))

        vel = self.vehicle.get_velocity()
        kmph = int(3.6 * math.sqrt(vel.x**2 + vel.y**2))

    # Reward Conditions

        if len(self.collision_hist) != 0:
            reward = -500
            done = True

        elif kmph < 50:
            done = False
            reward = -1

        elif kmph > 85:
            done = False
            reward = -10

        elif 50 <= kmph <= 85:
            done = False
            reward = 10

        if self.episode_start + EPISODE_LENGTH_IN_SECONDS < time.time():
            done = True

        return self.front_camera, reward, done, None

    # finally:
    #     print('Destroying everything')
    #     rgb_camera.destroy()
    #     client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    #     print('Done!')


