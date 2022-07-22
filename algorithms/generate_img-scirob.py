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
from self_cancel_fn3 import *
from bounding_box_fn2 import *
from continuity_fn import *
import matplotlib.pylab as plt

print('generating CNN dataset')

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
        reset_to_initial(0, reset_array_raw, client, vehicle_name="drone0")
        old_posit["drone0"] = client.simGetVehiclePose(vehicle_name="drone0")
        reset_to_initial(0, reset_array_raw, client, vehicle_name="drone1")
        old_posit["drone1"] = client.simGetVehiclePose(vehicle_name="drone1")
        print('current position', old_posit[name_agent])

        env_cfg = read_cfg(config_filename=env_folder+'config.cfg')
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z,f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client, env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

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
    video_L = cv2.VideoWriter('optical_L4.avi',fourcc,30,(1500,1000))
    video_R = cv2.VideoWriter('optical_R4.avi',fourcc,30,(1500,1000))
    video_event = cv2.VideoWriter('dvs3.avi',fourcc,10,(3000,2000),0)
    #video_event = cv2.VideoWriter('dvs3.avi',fourcc,10,(1500,1000),0)
    video_optical = cv2.VideoWriter('optical3.avi',fourcc,10,(1500,1000))
    # last_crash_array = np.zeros(shape=len(level_name), dtype=np.int32)
    # ret_array = np.zeros(shape=len(level_name))
    # distance_array = np.zeros(shape=len(level_name))
    # epi_env_array = np.zeros(shape=len(level_name), dtype=np.int32)
    max_speed = 0.002
    yaw_prev = 0.
    offset_x = reset_array_raw['drone1'][0][0]-reset_array_raw['drone0'][0][0] 
    offset_y = reset_array_raw['drone1'][0][1]-reset_array_raw['drone0'][0][1] 
    Vel = [] #np.zeros((1500,4))
    c = Core()
    c.set_model(c.get_model())
    prev_image = np.zeros((1000,1500,3))
    position_prev = [0,0]

    lin1 = np.append(np.linspace(1,0,500),np.linspace(0,1,500))
    temp1 = np.expand_dims(lin1,axis = 1)
    temp2 = np.ones((1,1500))
    vert_cancel_array = temp1*temp2

    lin1 = np.append(np.linspace(1,0,750),np.linspace(0,1,750))
    temp1 = np.expand_dims(lin1,axis = 0)
    temp2 = np.ones((1000,1))
    horz_cancel_array = temp2*temp1

    one_array = np.ones((1000,1500))

    count = 0
    vz = 0
    vy = 0
    vx = 0
    yaw_speed = 0
    window = 1
    area = 4000
    frequency = 5
    depth = np.zeros((1000,1500,3))
    V = np.zeros((1000, 1500))
    print('offsets',offset_x,offset_y)
    valid_drone_x = -offset_x
    valid_drone_y = -offset_y
    v_x = 0
    v_y = 0
    found = 0
    
    
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
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array([old_posit[name_agent].position.x_val-posit[name_agent].position.x_val,old_posit[name_agent].position.y_val-posit[name_agent].position.y_val]))
                        #altitude[name_agent].append(-posit[name_agent].position.z_val+p_z)
                        #altitude[name_agent].append(-posit[name_agent].position.z_val-f_z)

                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val, posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val
                        
                        if name_agent == "drone1":
                            print()
                            current_state[name_agent] = agent[name_agent].get_state1()
                            action, action_type, algorithm_cfg.epsilon, qvals, step = policy(1, current_state[name_agent], iter,
                                                                              algorithm_cfg.epsilon_saturation, 'inference',
                                                                              algorithm_cfg.wait_before_train, algorithm_cfg.num_actions, agent[name_agent],action_array, step)
                            action_word = translate_action(action, algorithm_cfg.num_actions)
                            
                            agent[name_agent].take_action(action, iter, algorithm_cfg.num_actions, SimMode=cfg.SimMode)
                            old_posit[name_agent] = posit[name_agent]
                            

                        else:
                            iter = iter + 1
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)], vehicle_name= name_agent)
                            response = responses[0]
                            depth = np.array(response.image_data_float, dtype=np.float32)
                            depth = depth.reshape(response.height, response.width)
                            depth = np.array(depth * 255, dtype=np.uint8)
                            start_time = time.time()
                            #print('depth',np.shape(depth))
                            

                            camera_image1 = camera_image.astype(float)
                            
                            dvs_image = abs(camera_image1[:,:,0] - prev_image[:,:,0])
                            dvs_image = np.sign(dvs_image - 80)
                            dvs_image = (dvs_image + 1)/2
                            
                            #print(dvs_image)
                            #plt.imshow(dvs_image)
                            #plt.show()
                            #input()    
                            prev_image = camera_image1
                            '''
                            active = np.nonzero((camera_image1[:,:,0] > 0.9))
                            y = active[1] #np.floor((active[0]/1080))+1.
                            x = active[0] #]np.mod(active[1], 1080)+1.
                            x = np.expand_dims(x, axis=0)
                            y = np.expand_dims(y, axis=0)

                            filtered = np.append(x,y, axis = 0)
                            '''
                            #print(np.shape(camera_image1))
                            dist_prey = 4.5/60* math.sqrt(area) ## calculated constant
                            
                            if found == 0:
                                yaw_speed = -0.0005
                            print('found',found)
                            yaw_speed = 0.0005*(v_x/500)/ dist_prey
                            v_up = (v_y/750) /dist_prey
                            v_forward = 0.01*dist_prey
                            #print('mine',v_forward, v_up, yaw_speed)
                            #v_forward = vx*math.cos(yaw)+vy*math.sin(yaw)
                            #v_sideward = vx*math.sin(yaw) - vy*math.cos(yaw)
                            
                            v_fh = (v_forward) * horz_cancel_array
                            v_fv = (v_forward) * vert_cancel_array
                            
                            vertical = 100*(abs(vz)+0.003)*one_array + 50*v_fv
                            horizontal = 30000*yaw_speed*one_array  + 100*v_fh # + 1000*0.8*v_sideward*one_array
                            velocity = np.append(np.abs(vertical),np.abs(horizontal),axis = 1) #(vz+0.003)*6500,yaw_speed*50000]
                            
                            [V, drone_x, drone_y, V3] = self_cancel_fn3(velocity, depth, dvs_image)
                            #V3 = np.expand_dims(V3, axis=0)
                            V3 = (255.*V3).astype(np.uint8)
                            #print(np.shape(V3))
                            
                            video_event.write(V3)
                            #print(V)
                            #plt.imshow((255.*V).astype(np.uint8))
                            #plt.show()
                            #input()
                            '''
                            for g in range(1, np.shape(cancelled)[1]):
                                V[int(cancelled[0,int(g)-1])-1,int(cancelled[1,int(g)-1])-1] = 1.
                            plt.imshow((255.*V).astype(np.uint8))
                            plt.show()
                            input()
                            '''

                            #[drone_x, drone_y] = bounding_box_fn2(V, window)
                            [drone_x, drone_y, count] = continuity_fn(drone_x, drone_y, position_prev, count)
                            #print('SNN identified coarse',drone_x,drone_y)
                            if (drone_x == 0 & drone_y == 0):
                                drone_x = valid_drone_x
                                drone_y = valid_drone_y
                            else:
                                valid_drone_x = drone_x
                                valid_drone_y = drone_y
                            #print("1 DVS creation %s seconds ---" % )
                            Vel.append([iter, valid_drone_x, valid_drone_y, time.time() - start_time, found, 0]) ## SNN is 0
                                #print('SNN identified',valid_drone_x,valid_drone_y)
                                #print('SNN identified',valid_drone_x*10,valid_drone_y*10)   
                            #print('snn position1', drone_x- 1500./2 , 1000./2 - drone_y)
                            
                            #print('snn position2' , drone_x, drone_y)
                            if iter%frequency == 1:
                                #print('iter', iter)
                            
                                drawing_image = c.get_drawing_image(camera_image)
                                processed_image, scale = c.pre_process_image(camera_image)
                                
                                boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)
                                box = boxes[0,0,:]
                                score = scores[0,0]
                                label = labels[0,0]
                                #print(box,label,score)
                                detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
                                #print(drawing_image)
                                for repeat in range(0,frequency):
                                    video_optical.write(drawing_image)

                                del_z = 0
                                #del_x = 0#-1*offset_x
                                #del_y = 0#-1*offset_y
                                #yaw_speed = 0.0008
                                found = 0
                                if label == 0:
                                    if score > 0.15:
                                        found = 1
                                        area = (box[2] - box[0]) * (box[3] - box[1])
                                        drone_x = box[0] + (box[2] - box[0])/2.
                                        drone_y = box[1] + (box[3] - box[1])/2.
                                        valid_drone_x = drone_x
                                        valid_drone_y = drone_y
                                        print('CNN identified',valid_drone_x,valid_drone_y, area)
                                        #print('cnn position1' , area, drone_x- 1500./2 , 1000./2 - drone_y)
                                Vel.append([iter, valid_drone_x, valid_drone_y, time.time() - start_time, found, 1])

                            if area > 4000:
                                frequency = 5
                            if area < 4000:
                                frequency = 2

                            v_x = valid_drone_x - 1500./2 
                            v_y = - valid_drone_y + 1000./2
                            

                            dist_prey = 4.5/60* math.sqrt(area) ## calculated constant
                            yaw_speed = 0.0005*(v_x/500) * dist_prey * 0.01
                            v_up = 0.01 * (v_y/750) * dist_prey
                            v_forward = 0.01*dist_prey
                            
                            if found == 0:
                                yaw_speed = -0.0005

                            vz = v_up*2
                            vx = v_forward * math.cos(yaw_prev)/4
                            vy = v_forward * math.sin(yaw_prev)/4
                            print('my',v_forward, v_up - 0.001 , yaw_speed)

                            if abs(vx) < max_speed:
                                vx = vx
                            else:
                                vx = max_speed  * np.sign(vx)

                            if abs(vy) < max_speed:
                                vy = vy
                            else:
                                vy = max_speed  * np.sign(vy)

                            if abs(vz) < max_speed:
                                vz = - vz - 0.001
                            else:
                                vz = -1*max_speed  * np.sign(vz) - 0.001
                            
                            #print(del_x, offset_x,del_y ,offset_y,del_z)
                            #print('velocity',vx,vy,vz,yaw_speed)
                            #print(yaw_speed*50000, 10000*(vx*math.sin(yaw)+vy*math.cos(yaw)), 10000*(vx*math.cos(yaw)+vy*math.sin(yaw)))
                            
                            
                            #print('velocity',vx,vy,vz,yaw_speed)
                            current_state[name_agent] = agent[name_agent].get_state(video_L,video_R)
                            
                            #action, action_type, algorithm_cfg.epsilon, qvals, step = policy(1, current_state[name_agent], iter,
                            #                                                  algorithm_cfg.epsilon_saturation, 'inference',
                            #                                                  algorithm_cfg.wait_before_train, algorithm_cfg.num_actions, agent[name_agent],action_array, step)
                            
                            #action_word = translate_action(action, algorithm_cfg.num_actions)
                            '''vx = 0.003
                            vy = 0 #0.003
                            vz = -0.003
                            yaw_speed = 0
                            '''

                            yaw_prev = agent[name_agent].take_action2(action, algorithm_cfg.num_actions,vx,vy,vz, yaw_speed,offset_x,offset_y, SimMode=cfg.SimMode)
                            #yaw_prev = agent[name_agent].take_action2(action, algorithm_cfg.num_actions,0.1,0,0,del_x,del_y,del_z, yaw_speed,offset_x,offset_y, SimMode=cfg.SimMode)
                            old_posit[name_agent] = posit[name_agent]
                            #input()
                            #Vel[iter][:] = [vx,vy,vz,yaw_prev]

                            if iter == 160:
                                Vel = np.array(Vel)
                                print(np.shape(Vel))
                                print(Vel)
                                np.savetxt('test.csv', Vel, delimiter=',')

                        # Verbose and log making
                    s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                        x_val, y_val, z_val, yaw, action_word
                    )

                    #print(s_log)
                    log_files.write(s_log + '\n')



        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)

                agent[name_agent].client = client
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')


