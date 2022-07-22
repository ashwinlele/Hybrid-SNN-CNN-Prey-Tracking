# Author: Aqeel Anwar(ICSRL)
# Created: 2/19/2020, 8:39 AM
# Email: aqeel.anwar@gatech.edu

import sys, cv2
#import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg
import csv
from core import Core
from self_cancel_fn4 import *
from bounding_box_fn2 import *
from continuity_fn import *
import matplotlib.pylab as plt
import matplotlib.patches as patches
from v2e_function import *


########## slomo libraries ##########
from v2e.slomo import SuperSloMo
import glob
import argparse
import importlib
from pathlib import Path
import os

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory, TemporaryFile
from engineering_notation import EngNumber  as eng # only from pip
from tqdm import tqdm


import v2e.desktop as desktop
from v2e.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2e.v2e_args import v2e_args, write_args_info, v2e_check_dvs_exposure_args
from v2e.v2e_args import NO_SLOWDOWN
from v2e.renderer import EventRenderer, ExposureMode
from v2e.slomo import SuperSloMo
from v2e.emulator import EventEmulator
from v2e.v2e_utils import inputVideoFileDialog
import logging

print('generating CNN dataset')

def frame_event_gen(frame1,frame2):
    frame1 = np.where(frame1<5,frame1/5*math.log(5),frame1)
    frame1 = np.where(frame1>=5,math.log(frame1),frame1)

    frame2 = np.where(frame2<5,frame2/5*math.log(5),frame2)
    frame2 = np.where(frame2>=5,math.log(frame2),frame2)
    return event_frame

def generate_img(cfg, env_process, env_folder):

    algorithm_cfg = read_cfg(config_filename='configs/generate_img.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    #if 'GlobalLearningGlobalUpdate-SA' in algorithm_cfg.distributed_algo:
    # algorithm_cfg = update_algorithm_cfg(algorithm_cfg, cfg)
    #cfg.num_agents = 1

    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)
    initial_pos = old_posit.copy()
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)
    print(reset_array_raw,'yyyy',cfg.num_agents)
    # Initialize System Handlers
    process = psutil.Process(getpid())
    # nvidia_smi.nvmlInit()

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    name_agent_list = []
    agent = {}
    # Replay Memory for RL
    if cfg.mode == 'train':
        ReplayMemory = {}
        target_agent = {}

        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
            print_orderly('global', 40)
            # Multiple agent acts as data collecter and one global learner
            global_agent = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name='global')
            ReplayMemory = Memory(algorithm_cfg.buffer_len)
            target_agent = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name='global')

        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)

            name_agent_list.append(name_agent)
            print_orderly(name_agent, 40)
            # TODO: turn the neural network off if global agent is present
            agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)

            if algorithm_cfg.distributed_algo != 'GlobalLearningGlobalUpdate-MA':
                ReplayMemory[name_agent] = Memory(algorithm_cfg.buffer_len)
                target_agent[name_agent] = PedraAgent(algorithm_cfg, client, name= 'Target', vehicle_name = name_agent)
            current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)
            print(name_agent,'iiiiiiiiiiii')
            name_agent_list.append(name_agent)
            agent[name_agent] = PedraAgent(algorithm_cfg, client, name = name_agent + 'DQN', vehicle_name = name_agent)
            print('agent',agent)
        
            env_cfg = read_cfg(config_filename=env_folder+'config.cfg')
            nav_x = []
            nav_y = []
            altitude = {}
            altitude[name_agent] = []
            p_z,f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client, env_folder=env_folder)
            nav_text = ax_nav.text(0, 0, '')

        reset_to_initial(0, reset_array_raw, client, vehicle_name="drone0")
        old_posit["drone0"] = client.simGetVehiclePose(vehicle_name="drone0")
        print('current position', old_posit["drone0"])
        reset_to_initial(0, reset_array_raw, client, vehicle_name="drone1")
        old_posit["drone1"] = client.simGetVehiclePose(vehicle_name="drone1")
        print('current position', old_posit["drone1"])
        
        # Select initial position
        # Initialize variables
    
    iter = 1
    step = 0
    action_array = 4*np.ones((10))
    action_array = action_array.astype(int)

    # num_collisions = 0
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array ={}
    ret_array ={}
    distance_array ={}
    epi_env_array = {}
    log_files = {}

    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    ignore_collision = False
    
    for name_agent in name_agent_list:
        #print(name_agent, name_agent_list)
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        distance[name_agent] = 0
        # Log file
        log_path = 'algorithms/log.txt'
        print('oooooo',log_path)
        #print("Log path: ", log_path)
        log_files = open(log_path, 'w')
        #print('state',agent[name_agent].get_state())

    # switch_env = False

    print_orderly('Simulation begins', 80)
    '''
    for itera in range(0,5):
        print('itera',itera)
        for name_agent in name_agent_list:
            x = itera*10
            y = itera*10
            z = -itera*10 #-1*itera*4
            
            alpha = itera*5
            psi = 10
            current_state[name_agent] = agent[name_agent].get_state()
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0,0, 20)),
                                     ignore_collison=ignore_collision, vehicle_name = 'drone0')
    '''
    
    print_orderly('Simulation begins', 80)
    

    # save_posit = old_posit
    iter = -1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_L = cv2.VideoWriter('optical_L4.mp4',fourcc,30,(640,480))
    video_R = cv2.VideoWriter('optical_R4.mp4',fourcc,30,(640,480))
    video_event = cv2.VideoWriter('dvs3.mp4',fourcc,10,(640*3,480*2),0)
    video_dvs = cv2.VideoWriter('optical_dvs.mp4',fourcc,10,(640,480))
    #video_event = cv2.VideoWriter('dvs3.avi',fourcc,10,(640,480),0)
    video_optical = cv2.VideoWriter('optical3.mp4',fourcc,10,(640,480))
    # last_crash_array = np.zeros(shape=len(level_name), dtype=np.int32)
    # ret_array = np.zeros(shape=len(level_name))
    # distance_array = np.zeros(shape=len(level_name))
    # epi_env_array = np.zeros(shape=len(level_name), dtype=np.int32)
    max_speed = 0.004
    yaw_prev = 0.
    offset_x = reset_array_raw['drone1'][0][0]-reset_array_raw['drone0'][0][0] 
    offset_y = reset_array_raw['drone1'][0][1]-reset_array_raw['drone0'][0][1] 
    Vel = np.zeros((200,31))
    c = Core()
    c.set_model(c.get_model())
    prev_image = np.zeros((480,640,3))
    position_prev = [0,0]
    dvs_image = np.zeros((480,640))
    lin1 = np.append(np.linspace(1,0,int(480/2)),np.linspace(0,1,int(480/2)))
    temp1 = np.expand_dims(lin1,axis = 1)
    temp2 = np.ones((1,640))
    vert_cancel_array = temp1*temp2

    lin1 = np.append(np.linspace(1,0,int(640/2)),np.linspace(0,1,int(640/2)))
    temp1 = np.expand_dims(lin1,axis = 0)
    temp2 = np.ones((480,1))
    horz_cancel_array = temp2*temp1

    one_array = np.ones((480,640))

    count = 0
    vz = 0
    vy = 0
    vx = 0
    yaw_speed = 0
    depth = np.zeros((480,640,3))
    V = np.zeros((480, 640))
    print('offsets',offset_x,offset_y)
    valid_drone_x = -offset_x
    valid_drone_y = -offset_y
    found = 0
    cnn_position = [0,0]
    snn_position = [0,0]
    
    v_forward = 0
    window = 10
    area = 10000
    frequency = 50
    found = 0
    cnn_position = [0,0]
    snn_position_prev = np.array([0,0]) 
    found_cnn = 0
    found_snn = 0
    suspect = 0
    suspect_threshold = 6
    snn_time = 0
    cnn_time = 0
    gt_y = 0
    gt_x = 0

    def get_args():
        parser = argparse.ArgumentParser(
            description='v2e: generate simulated DVS events from video.',
            epilog='Run with no --input to open file dialog', allow_abbrev=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = v2e_args(parser)
        parser.add_argument(
            "--rotate180", type=bool, default=False,
            help="rotate all output 180 deg.")
        # https://kislyuk.github.io/argcomplete/#global-completion
        # Shellcode (only necessary if global completion is not activated -
        # see Global completion below), to be put in e.g. .bashrc:
        # eval "$(register-python-argcomplete v2e.py)"
        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        return args
    
    args=get_args()
    slowdown_factor=NO_SLOWDOWN
    output_folder: str = os.getcwd()  + "\\output\\tennis\\"
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview = False
    batch_size = 4
    
    slomo = SuperSloMo(
                model=os.getcwd()+"\\input\\SuperSloMo39.ckpt", auto_upsample=False, upsampling_factor=10,
                video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo,
                preview=preview, batch_size=batch_size)
    
    while active:
        try:
            active, automate, algorithm_cfg, client = check_user_input(active, automate, agent[name_agent], client, old_posit[name_agent], initZ, fig_z, fig_nav, env_folder, cfg, algorithm_cfg)
            #print()
            if automate:

                
                if cfg.mode == 'infer':
                    for drone in range(cfg.num_agents):
                        drone1 = 1 - drone
                        name_agent = "drone" + str(drone1)
                        #print(name_agent,'bbbb')
                        # Inference phase
                        agent_state = agent[name_agent].GetAgentState()
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        #print('position',posit[name_agent])
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array([old_posit[name_agent].position.x_val-posit[name_agent].position.x_val,old_posit[name_agent].position.y_val-posit[name_agent].position.y_val]))
                        #altitude[name_agent].append(-posit[name_agent].position.z_val+p_z)
                        #altitude[name_agent].append(-posit[name_agent].position.z_val-f_z)

                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val, posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val
                        
                        if name_agent == "drone1":
                            print('here')
                            current_state[name_agent] = agent[name_agent].get_state1()
                            action, action_type, algorithm_cfg.epsilon, qvals, step = policy(1, current_state[name_agent], iter,
                                                                              algorithm_cfg.epsilon_saturation, 'inference',
                                                                              algorithm_cfg.wait_before_train, algorithm_cfg.num_actions, agent[name_agent],action_array, step)
                            action_word = translate_action(action, algorithm_cfg.num_actions)
                            
                            prey_speed = agent[name_agent].take_action(action, iter, algorithm_cfg.num_actions, SimMode=cfg.SimMode)
                            old_posit[name_agent] = posit[name_agent]
                            

                        else:
                            iter = iter + 1
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))
                            
                            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)], vehicle_name= name_agent)
                            response = responses[0]
                            depth = np.array(response.image_data_float, dtype=np.float32)
                            depth = depth.reshape(response.height, response.width)
                            depth = np.array(depth * 255, dtype=np.uint8)
                            depth = depth.astype(float)/255
                            depth = cv2.resize(depth,(640,480))

                            camera_image1 = camera_image.astype(float)
                            if iter > 2:
                                dvs_image = v2e_function(slomo)
                                
                            video_dvs = cv2.VideoWriter('optical_dvs.avi',fourcc,10,(640,480))
                            
                            prev_image = camera_image1
                            
                            del_x = posit["drone1"].position.x_val - posit[name_agent].position.x_val + offset_x
                            del_y = posit["drone1"].position.y_val - posit[name_agent].position.y_val + offset_y
                            del_z = posit["drone1"].position.z_val - posit[name_agent].position.z_val
                            dist = math.sqrt(del_x**2 + del_y**2) 
                            
                            binary = np.where(del_x > 0, 1, 0)
                            angle_now = binary*np.arctan(del_y/del_x) + (1 - binary)*(np.arctan(del_y/del_x) + 3.14)

                            angle_now2 = math.atan2(del_z,dist)
                            if abs(angle_now - yaw_prev) < 3.14/2 and angle_now2 <10 :
                                gt_x = int(640/2 + 1074*(angle_now - yaw_prev))   # constant = 40 degrees map to 750 pixels
                                gt_y = int(480/2 + 1074*angle_now2)
                                size = 10000/dist
                            
                            v_fh = (v_forward) * horz_cancel_array 
                            v_fv = (v_forward) * vert_cancel_array
                            
                            iteration_begin_time = time.time()
                            vertical = 0 + 30*v_fv #100*(abs(vz)+0.003)*one_array
                            horizontal = 250*yaw_speed*one_array  + 30*v_fh # + 1000*0.8*v_sideward*one_array
                            velocity = np.append(5*np.abs(vertical), 5*np.abs(horizontal), axis = 1) #(vz+0.003)*6500,yaw_speed*50000]
                            
                            start_time = time.time()
                            [V, drone_x, drone_y, V3, count] = self_cancel_fn4(velocity, depth, dvs_image, snn_position_prev, count)
                            snn_time = time.time() - start_time
                            
                            found_snn = 0
                            found_snn_act = 0
                            if suspect > suspect_threshold + 1:
                                suspect = 0
                            if (drone_x == 0 & drone_y == 0):
                                found_snn = 0
                                #suspect = 0
                            else:
                                snn_position = np.array([drone_x,drone_y])
                                print('snn_position',snn_position)
                                snn_position_prev = snn_position
                                if found_cnn == 0:
                                    if np.sum(((snn_position-snn_position_prev)**2)) < 15000.:
                                        suspect = suspect + 1
                                        found_snn = 1
                                        if suspect > suspect_threshold:
                                            found_snn_act = 1
                                            valid_drone_x = snn_position[0]
                                            valid_drone_y = snn_position[1]
                                        
                                    else:
                                        suspect = max(suspect - 1,0)
                                        found_snn_act = 0
                                        found_snn = 0
                                else:
                                    if np.sum(((snn_position-cnn_position)**2)) < 15000.:
                                        #suspect = suspect + 1
                                        found_snn = 1
                                        valid_drone_x = snn_position[0]
                                        valid_drone_y = snn_position[1]
                                    else:
                                        suspect = max(suspect - 1,0)


                            V3 = (255.*V3).astype(np.uint8)
                            video_event.write(V3)
                            

                            found = found_snn
                            if suspect == suspect_threshold or iter% frequency == 0:
                                print("CNN")

                                start_time = time.time()
                                camera_image1 = camera_image[:,:,[2,1,0]]
                                drawing_image = c.get_drawing_image(camera_image1)
                                processed_image, scale = c.pre_process_image(camera_image1)
                                boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
                                box = boxes[0,0,:]
                                score = scores[0,0]
                                label = labels[0,0]
                                detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
                                found_cnn = 0
                                cnn_position = [0,0]
                                del_z = 0
                                if label == 0:
                                    if score > 0.15:
                                        area = (box[2] - box[0]) * (box[3] - box[1])
                                        drone_x = box[0] + (box[2] - box[0])/2.
                                        drone_y = box[1] + (box[3] - box[1])/2.
                                        valid_drone_x = drone_x
                                        valid_drone_y = drone_y
                                        cnn_position = [drone_x,drone_y]
                                        found_cnn = 1
                                found = found_cnn
                                video_optical.write(drawing_image.astype(np.uint8))
                                cnn_time = time.time() - start_time
                                
                            if area < 9000:
                                frequency = 8 ## put twice the vlue that is needed
                            else:
                                frequency = 16 

                            
                            v_x = valid_drone_x - 640./2 
                            v_y = - valid_drone_y + 480./2
                            
                            
                            dist_prey = 1/math.sqrt(area) ## calculated constant
                            yaw_speed = 0.009*(v_x/500) * math.sqrt(area)/20
                            v_up = 0 #0.01 * (v_y/7500) * dist_prey
                            v_forward = 1*dist_prey

                            vz = v_up*2
                            vx = v_forward * math.cos(yaw_prev)/4
                            vy = v_forward * math.sin(yaw_prev)/4
                            
                            if (found_cnn ==0 and found_snn_act == 0):
                                yaw_speed = 0.003
                                vz = -0.00305 
                                vx = 0
                                vy = 0
                            else:
                                if abs(vx) < max_speed:
                                    vx = vx
                                else:
                                    vx = max_speed  * np.sign(vx)

                                if abs(vy) < max_speed:
                                    vy = vy
                                else:
                                    vy = max_speed  * np.sign(vy)

                                vz = - 0.00305  
                                
                            if (math.sqrt((snn_position[0] - gt_x)**2 + (snn_position[1] - gt_y)**2)) < 20000:
                                snn_accuracy = 1
                            else:
                                snn_accuracy = 0
                            
                            if (math.sqrt((cnn_position[0] - gt_x)**2 + (cnn_position[1] - gt_y)**2)) < 20000: # within 100 x 100 pixel window
                                cnn_accuracy = 1
                            else:
                                cnn_accuracy = 0

                            
                            current_state[name_agent] = agent[name_agent].get_state(video_L,video_R)
                            yaw_prev = agent[name_agent].take_action2(action, algorithm_cfg.num_actions,vx,vy,vz, yaw_speed,offset_x,offset_y, SimMode=cfg.SimMode)
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))

                            #yaw_prev = agent[name_agent].take_action2(action, algorithm_cfg.num_actions,0.1,0,0,del_x,del_y,del_z, yaw_speed,offset_x,offset_y, SimMode=cfg.SimMode)
                            old_posit[name_agent] = posit[name_agent]
                            print('motion',iter, found_snn, found_cnn, found, suspect, valid_drone_x, valid_drone_y, area, yaw_speed, vx,vy,vz)

                            x_val0 = posit['drone0'].position.x_val
                            y_val0 = posit['drone0'].position.y_val
                            z_val0 = posit['drone0'].position.z_val
                            x_val1 = posit['drone1'].position.x_val
                            y_val1 = posit['drone1'].position.y_val
                            z_val1 = posit['drone1'].position.z_val

                            Vel[iter,:] = [iter, found_snn, found_cnn, found, suspect, snn_time, cnn_time, valid_drone_x, valid_drone_y, area, yaw_speed, v_up, v_forward,x_val0,y_val0,z_val0,x_val1,y_val1,z_val1,cnn_position[0],cnn_position[1],snn_position[0],snn_position[1],yaw_prev,snn_accuracy,cnn_accuracy,gt_x,gt_y,time.time()-iteration_begin_time,prey_speed[0],prey_speed[1]]
                            #Vel.append([iter, valid_drone_x, valid_drone_y, time.time() - start_time, found, 0]) ## SNN is 0
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))
                            

                            if iter%20 == 1:
                                Vel = np.array(Vel)
                                np.savetxt('test.csv', Vel, delimiter=',')

                        # Verbose and log making
                    s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                        x_val, y_val, z_val, yaw, action_word
                    )

                    log_files.write(s_log + '\n')



        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)

                agent[name_agent].client = client
                video_dvs.release()
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')


