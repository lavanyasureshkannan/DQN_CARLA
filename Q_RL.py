#!/usr/bin/env python

import glob
import math
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Activation, GlobalAveragePooling2D
# from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.applications.xception import Xception
# from keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm

import tensorflow.keras.backend as backend

import numpy as np
import cv2
from collections import deque

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

MAX_REPLAY_SIZE = 5_000
MIN_REPLAY_SIZE = 1_000
MIN_BATCH_SIZE = 16
PREDICT_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = MIN_BATCH_SIZE // 4
UPDATE_TARGET = 4

NUM_EPISODES = 150
EPISODE_LENGTH_IN_SECONDS = 12

DISCOUNT = 0.95
EPSILON_DECAY = 0.98
EPSILON = 1
MIN_EPSILON = 0.001
MIN_REWARD = -200

MODEL_NAME = "Custom_Xception"
AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


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
        self.front_camera = im_rgb
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


class DQNModel:
    def __int__(self):
        self.model = self.dqn_model()
        self.target_model = self.dqn_model()
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.replay_memory = deque(maxlen=MAX_REPLAY_SIZE)

        # self.tensorboard = ModifiedTensorBoard(Log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False

        self.last_logged_episode = 0
        self.training_initialized = False

    def dqn_model(self):
        base = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        x = base.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base.input, outputs=predictions)
        model.compile(Loss="mse", optimizer=Adam(Lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory)< MIN_REPLAY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MIN_BATCH_SIZE)

        current_state = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_q_list = self.model.predict(current_state, PREDICT_BATCH_SIZE)

        new_current_state = np.array([transition[3] for transition in minibatch]) / 255
        with self.graph.as_default():
            future_q_list = self.target_model.predict(new_current_state, PREDICT_BATCH_SIZE)

        X = []
        y = []

        for index, (current, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_q_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_q = current_q_list[index]
            current_q[action] = new_q

            X.append(current_state)
            y.append(current_q)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter> UPDATE_TARGET:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        # self.graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    gpu_setting = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_setting)))

    if not os.path.isdir("models"):
        os.mkdir("models")

    agent = DQNModel()
    env = VehicleEnv()

    trainer_thread = Thread(target=agent.train_in_loop(), daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones(env.img_height, env.img_width, 3))

    for episode in tqdm(range(1, NUM_EPISODES+1), ascii=True, unit="episodes"):
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > EPSILON:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0,3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.action_step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            step += 1

            if done:
                break

        for actor in env.actor_list:
            print('Destroying everything')
            actor.destroy()

            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=EPSILON)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(MIN_EPSILON, EPSILON)

            # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')






# finally:
#     print('Destroying everything')
#     rgb_camera.destroy()
#     client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
#     print('Done!')