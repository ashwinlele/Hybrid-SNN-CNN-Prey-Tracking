#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
@latest update: Apr 2020
"""
# todo refractory period for pixel

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

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING))
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR))
logger = logging.getLogger(__name__)

# may only apply to windows

try:
    from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception:
    logger.warning('Gooey GUI builder not available, will use command line arguments.\n'
                   'Install with "pip install Gooey". See README')

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

def makeOutputFolder(output_folder_base, suffix_counter,
                     overwrite, unique_output_folder):
    if overwrite and unique_output_folder:
        logger.error("specify one or the other of --overwrite and --unique_output_folder")
        v2e_quit()
    if suffix_counter > 0:
        output_folder = output_folder_base + '-' + str(suffix_counter)
    else:
        output_folder = output_folder_base
    nonEmptyFolderExists = not overwrite and os.path.exists(output_folder) and os.listdir(output_folder)
    if nonEmptyFolderExists and not overwrite and not unique_output_folder:
        logger.error(
            'non-empty output folder {} already exists \n '
            '- use --overwrite or --unique_output_folder'.format(
                os.path.abspath(output_folder), nonEmptyFolderExists))
        v2e_quit()

    if nonEmptyFolderExists and unique_output_folder:
        return makeOutputFolder(
            output_folder_base, suffix_counter + 1, overwrite, unique_output_folder)
    else:
        logger.info('using output folder {}'.format(output_folder))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder


def v2e_function(): #main():
    try:
        ga=Gooey(get_args, program_name="v2e", default_size=(575, 600))
        logger.info('Use --ignore-gooey to disable GUI and run with command line arguments')
        ga()
    except:
        logger.warning('Gooey GUI not available, using command line arguments. \n'
                       'You can try to install with "pip install Gooey"')
    args=get_args()
    output_width = 640 #: int = args.output_width
    output_height = 480 # int = args.output_height
    if (output_width is None) ^ (output_height is None):
        logger.error('set neither or both of output_width and output_height')
        v2e_quit()


    synthetic_input= None
    synthetic_input_module=None
    synthetic_input_class=None
    synthetic_input_instance=None
    synthetic_input_next_frame_method=None
    if synthetic_input is not None:
        try:
            synthetic_input_module = importlib.import_module(synthetic_input)
            synthetic_input_class = getattr(synthetic_input_module,synthetic_input)
            synthetic_input_instance=synthetic_input_class(width=output_width, height=output_height,preview=not args.no_preview)
            synthetic_input_next_frame_method=getattr(synthetic_input_class,'next_frame')

        except ModuleNotFoundError as e:
            logger.error(f'Could not import {synthetic_input}: {e}')
            quit(1)
        except AttributeError as e:
            logger.error(f'{synthetic_input} method incorrect?: {e}')
            quit(1)

    input_file = os.getcwd()  + "\\optical_dvs.avi" #args.input
    if synthetic_input is None and not input_file:
        input_file = inputVideoFileDialog()
        if not input_file:
            v2e_quit()

    overwrite: bool = True
    output_folder: str = os.getcwd()  + "\\output\\tennis\\" #args.output_folder # #
    unique_output_folder: bool = args.unique_output_folder if not overwrite else False # if user specifies overwrite, then override default of making unique output folder
    output_in_place: bool=args.output_in_place if (not synthetic_input and output_folder is None) else False
    num_frames=0
    srcNumFramesToBeProccessed=0

    if output_in_place:
        parts=os.path.split(input_file)
        output_folder=parts[0]
    else:
        output_folder = makeOutputFolder(output_folder, 0, overwrite, unique_output_folder)


    # input file checking
    if (not input_file or not Path(input_file).exists()) and not synthetic_input:
        v2e_quit()

    start_time = args.start_time
    stop_time = 3 #args.stop_time

    input_slowmotion_factor:float = args.input_slowmotion_factor
    timestamp_resolution:float = 0.003 #args.timestamp_resolution
    auto_timestamp_resolution:bool=  False #args.auto_timestamp_resolution
    disable_slomo:bool= False #True
    slomo=None # make it later on

    
    pos_thres = 0.15
    neg_thres = 0.15
    sigma_thres = 0.03
    cutoff_hz = 15
    leak_rate_hz = args.leak_rate_hz
    
    shot_noise_rate_hz = args.shot_noise_rate_hz
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = "tennis.aedat" #args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    slomo_stats_plot=args.slomo_stats_plot

    preview = not args.no_preview
    rotate180 = args.rotate180
    batch_size = 4 #args.batch_size

    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)
    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1. / exposure_val

    infofile = write_args_info(args, output_folder)

    fh = logging.FileHandler(infofile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    import time
    time_run_started = time.time()

    slomoTimestampResolutionS = None
    if synthetic_input is None:
        
        cap = cv2.VideoCapture(input_file)
        srcFps = cap.get(cv2.CAP_PROP_FPS)
        if srcFps == 0:
            logger.error('source {} fps is 0; v2e needs to have a timescale for input video'.format(input_file))
            v2e_quit()

        srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if srcNumFrames < 2:
            logger.warning(
                'num frames is less than 2, probably cannot be determined '
                'from cv2.CAP_PROP_FRAME_COUNT')

        srcTotalDuration = (srcNumFrames - 1) / srcFps
        start_frame = int(srcNumFrames * (start_time / srcTotalDuration)) \
            if start_time else 0
        stop_frame = int(srcNumFrames * (stop_time / srcTotalDuration)) \
            if stop_time else srcNumFrames
        srcNumFramesToBeProccessed = stop_frame - start_frame + 1
        srcDurationToBeProcessed = srcNumFramesToBeProccessed / srcFps
        start_time = start_frame / srcFps
        stop_time = stop_frame / srcFps  # todo something replicated here, already have start and stop times

        srcFrameIntervalS = (1. / srcFps) / input_slowmotion_factor


        slowdown_factor=NO_SLOWDOWN # start with factor 1 for upsampling
        if disable_slomo:
            logger.info('slomo interpolation disabled by command line option; output DVS timestamps will have source frame interval resolution')
        elif not auto_timestamp_resolution:
            slowdown_factor=int(np.ceil(srcFrameIntervalS/timestamp_resolution))
            if slowdown_factor < NO_SLOWDOWN:
                slowdown_factor = NO_SLOWDOWN
            slomoTimestampResolutionS = srcFrameIntervalS / slowdown_factor
        
            check_lowpass(cutoff_hz, 1 / slomoTimestampResolutionS, logger)
        
        # the SloMo model, set no SloMo model if no slowdown
        if not disable_slomo and ( auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN ):
            slomo = SuperSloMo(
                model=args.slomo_model, auto_upsample=auto_timestamp_resolution, upsampling_factor=slowdown_factor,
                video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo,
                preview=preview, batch_size=batch_size)

    if not synthetic_input and not auto_timestamp_resolution:
        slomoTimestampResolutionS = srcFrameIntervalS / 1
        #logger.info(f'\n events will have timestamp resolution {eng(slomoTimestampResolutionS)}s,')

    if not synthetic_input:
        
        if exposure_mode == ExposureMode.DURATION:
            dvsNumFrames = np.math.floor(
                dvsFps * srcDurationToBeProcessed / input_slowmotion_factor)
            dvsDuration = dvsNumFrames / dvsFps
            dvsPlaybackDuration = dvsNumFrames / avi_frame_rate
            start_time = start_frame / srcFps
            stop_time = stop_frame / srcFps  # todo something replicated here, already have start and stop times

    

    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2,
        dvs_text=dvs_text, show_dvs_model_state=args.show_dvs_model_state)

    if args.dvs_params:
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension)

    if synthetic_input_next_frame_method is not None:
        events = np.zeros((0, 4), dtype=np.float32)  # array to batch events for rendering to DVS frames
        (fr,time)=synthetic_input_instance.next_frame()
        i=0
        with tqdm(total=synthetic_input_instance.total_frames(), desc='dvs', unit='fr') as pbar: # instantiate progress bar
            while fr is not None:
                newEvents = emulator.generate_events(fr, time)
                pbar.update(1)
                i+=1
                if newEvents is not None and newEvents.shape[0] > 0:
                    events = np.append(events, newEvents, axis=0)
                    events = np.array(events)
                    if i%batch_size==0:
                        eventRenderer.render_events_to_frames(events, height=output_height, width=output_width)
                        events = np.zeros((0, 4), dtype=np.float32)  # clear array
                (fr,time)=synthetic_input_instance.next_frame()
            if len(events)>0: # process leftover
                eventRenderer.render_events_to_frames(events, height=output_height, width=output_width)
    else: # file input
        # timestamps of DVS start at zero and end with span of video we processed
        srcVideoRealProcessedDuration = (stop_time - start_time) / input_slowmotion_factor
        num_frames = srcNumFramesToBeProccessed
        inputHeight = None
        inputWidth = None
        inputChannels = None
        if start_frame > 0:
            for i in tqdm(range(start_frame), unit='fr', desc='src'):
                ret, _ = cap.read()
                if not ret:
                    raise ValueError(
                        'something wrong, got to end of file before '
                        'reaching start_frame')

        with TemporaryDirectory() as source_frames_dir:
            inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            inputChannels = 1 if int(cap.get(cv2.CAP_PROP_MONOCHROME)) else 3

            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
            for inputFrameIndex in tqdm(
                    range(srcNumFramesToBeProccessed),
                    desc='rgb2luma', unit='fr'):
                # read frame
                ret, inputVideoFrame = cap.read()

                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

                if output_height and output_width and \
                        (inputHeight != output_height or
                         inputWidth != output_width):
                    dim = (output_width, output_height)
                    (fx, fy) = (float(output_width) / inputWidth,
                                float(output_height) / inputHeight)
                    inputVideoFrame = cv2.resize(
                        src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                        interpolation=cv2.INTER_AREA)
                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        logger.info(
                            '\nConverting input frames from RGB color to luma')
                    inputVideoFrame = cv2.cvtColor(
                        inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy")
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
            cap.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes=None
                # make input to slomo
                if slowdown_factor != NO_SLOWDOWN:
                    # interpolated frames are stored to tmpfolder as
                    # 1.png, 2.png, etc

                    logger.info(f'*** Stage 2/3: SloMo upsampling from {source_frames_dir}')
                    interpTimes,avgUpsamplingFactor=slomo.interpolate(
                        source_frames_dir, interpFramesFolder,
                        (output_width, output_height))
                    avgTs = srcFrameIntervalS / avgUpsamplingFactor
                    logger.info('SloMo average upsampling factor={:5.2f}; average DVS timestamp resolution={}s'
                                .format(avgUpsamplingFactor, eng(avgTs)))
                    # read back to memory
                    interpFramesFilenames = all_images(interpFramesFolder)
                    # number of frames
                    n = len(interpFramesFilenames)
                else:
                    logger.info(f'*** Stage 2/3:turning npy frame files to png from {source_frames_dir}')
                    interpFramesFilenames = []
                    n=0
                    src_files = sorted(
                        glob.glob("{}".format(source_frames_dir) + "/*.npy"))
                    for frame_idx, src_file_path in tqdm(
                            enumerate(src_files), desc='npy2png', unit='fr'):
                        src_frame = np.load(src_file_path)
                        tgt_file_path = os.path.join(
                            interpFramesFolder, str(frame_idx) + ".png")
                        interpFramesFilenames.append(tgt_file_path)
                        n+=1
                        cv2.imwrite(tgt_file_path, src_frame)
                    interpTimes=np.array(range(n))


                # compute times of output integrated frames
                nFrames=len(interpFramesFilenames)
                # interpTimes is in units of 1 per input frame, normalize it to src video time range
                f=srcVideoRealProcessedDuration/(np.max(interpTimes)-np.min(interpTimes))
                interpTimes = f*interpTimes # compute actual times from video times
                # debug
                if slomo_stats_plot:
                    from matplotlib import pyplot as plt  # TODO debug
                    dt = np.diff(interpTimes)
                    fig=plt.figure()
                    ax1=fig.add_subplot(111)
                    ax1.set_title('Slo-Mo frame interval stats (close to continue)')
                    ax1.plot(interpTimes)
                    ax1.plot(interpTimes,'x')
                    ax1.set_xlabel('frame')
                    ax1.set_ylabel('frame time (s)')
                    ax2=ax1.twinx()
                    ax2.plot(dt*1e3)
                    ax2.set_ylabel('frame interval (ms)')
                    logger.info('close plot to continue')
                    fig.show()

                events = np.zeros((0, 4), dtype=np.float32)  # array to batch events for rendering to DVS frames

                logger.info(f'*** Stage 3/3: emulating DVS events from {nFrames} frames')
                with tqdm(total=nFrames, desc='dvs', unit='fr') as pbar: # instantiate progress bar
                    for i in range(nFrames):
                        fr = read_image(interpFramesFilenames[i])
                        newEvents = emulator.generate_events(fr, interpTimes[i])
                        pbar.update(1)
                        if newEvents is not None and newEvents.shape[0] > 0:
                            events = np.append(events, newEvents, axis=0)
                            events = np.array(events)
                                
                            if i%batch_size==0:
                                break
                                eventRenderer.render_events_to_frames(events, height=output_height, width=output_width)
                                events = np.zeros((0, 4), dtype=np.float32)  # clear array
                    if len(events)>0: # process leftover
                        eventRenderer.render_events_to_frames(events, height=output_height, width=output_width)

    data = np.loadtxt(os.getcwd()+'\\output\\tennis\\v2e-dvs-events.txt')
    event_frame = 0.5*np.ones((480,640))
    event_frame[data[20000:40000,2].astype(int),data[20000:40000,1].astype(int)] = data[20000:40000,3] 
    cv2.imshow('events',event_frame)
    cv2.waitKey()
    #print(data[0:50,1])

    eventRenderer.cleanup()
    emulator.cleanup()
    if slomo is not None:
        slomo.cleanup()

    if num_frames == 0:
        logger.error('no frames read from file')
        v2e_quit()
    totalTime = (time.time() - time_run_started)
    framePerS = num_frames / totalTime
    sPerFrame = 1 / framePerS
    throughputStr = (str(eng(framePerS)) + 'fr/s') \
        if framePerS > 1 else (str(eng(sPerFrame)) + 's/fr')
    
    return event_frame
'''    
    try:
        desktop.open(os.path.abspath(output_folder))
    except Exception as e:
        logger.warning(
            '{}: could not open {} in desktop'.format(e, output_folder))
    

if __name__ == "__main__":
    main()
    v2e_quit()
'''