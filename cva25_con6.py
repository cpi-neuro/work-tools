#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Juli 10, 2025, at 17:18
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
from psychopy_bids.bids import BIDSBehEvent
from psychopy_bids.bids import BIDSTaskEvent
from psychopy_bids.bids import BIDSError
from psychopy_bids.bids import BIDSHandler

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'cva25_con6'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Carmen\\Desktop\\TDA\\cva25_con6.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_HTLF') is None:
        # initialise key_HTLF
        key_HTLF = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTLF',
        )
    if deviceManager.getDevice('key_HTLU') is None:
        # initialise key_HTLU
        key_HTLU = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTLU',
        )
    if deviceManager.getDevice('key_HTRF') is None:
        # initialise key_HTRF
        key_HTRF = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTRF',
        )
    if deviceManager.getDevice('key_HTRU') is None:
        # initialise key_HTRU
        key_HTRU = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTRU',
        )
    if deviceManager.getDevice('key_HTLF_OD') is None:
        # initialise key_HTLF_OD
        key_HTLF_OD = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTLF_OD',
        )
    if deviceManager.getDevice('key_HTRF_OD') is None:
        # initialise key_HTRF_OD
        key_HTRF_OD = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_HTRF_OD',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    if expInfo['session']:
        bids_handler = BIDSHandler(dataset='bids_cva25',
         subject=expInfo['participant'], task=expInfo['expName'],
         session=expInfo['session'], data_type='func', acq='',
         runs=True)
    else:
        bids_handler = BIDSHandler(dataset='bids_cva25',
         subject=expInfo['participant'], task=expInfo['expName'],
         data_type='func', acq='', runs=True)
    bids_handler.createDataset()
    bids_handler.addLicense('CC-BY-NC-4.0', force=True)
    bids_handler.addTaskCode(force=True)
    bids_handler.addEnvironment()
    
    # --- Initialize components for Routine "Trigger" ---
    # Run 'Begin Experiment' code from code_trigger
    from psychopy.hardware.emulator import launchScan
    
    v_MRinfo = { 'TR': 1.6,
                            'sync': '6',
                            'volumes':80, # 225
                            'sound': False,
                            'skip': 0}
    launchScan(win=win, settings=v_MRinfo, globalClock=core. Clock(), mode='scan', wait_msg='syncing')
    
    # --- Initialize components for Routine "HTLF" ---
    image1_HTLF = visual.ImageStim(
        win=win,
        name='image1_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTLF = visual.ImageStim(
        win=win,
        name='image2_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTLF = visual.ImageStim(
        win=win,
        name='image3_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTLF = visual.ImageStim(
        win=win,
        name='image4_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTLF = visual.ImageStim(
        win=win,
        name='image5_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTLF = visual.ImageStim(
        win=win,
        name='image6_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTLF = visual.ImageStim(
        win=win,
        name='image7_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTLF = visual.ImageStim(
        win=win,
        name='image8_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTLF = visual.ImageStim(
        win=win,
        name='image9_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTLF = visual.ImageStim(
        win=win,
        name='image10_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTLF = visual.ImageStim(
        win=win,
        name='image11_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTLF = visual.ImageStim(
        win=win,
        name='image12_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTLF = visual.ImageStim(
        win=win,
        name='image13_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTLF = visual.ImageStim(
        win=win,
        name='image14_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTLF = visual.ImageStim(
        win=win,
        name='image15_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTLF = visual.ImageStim(
        win=win,
        name='image16_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTLF = visual.ImageStim(
        win=win,
        name='image17_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTLF = visual.ImageStim(
        win=win,
        name='image18_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTLF = visual.ImageStim(
        win=win,
        name='image19_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTLF = visual.ImageStim(
        win=win,
        name='image20_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTLF = visual.ImageStim(
        win=win,
        name='image21_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTLF = visual.ImageStim(
        win=win,
        name='image22_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTLF = visual.ImageStim(
        win=win,
        name='image23_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTLF = visual.ImageStim(
        win=win,
        name='image24_HTLF', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTLF = visual.ImageStim(
        win=win,
        name='image_Frame_HTLF', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTLF = visual.ImageStim(
        win=win,
        name='image_Cue_HTLF', 
        image='images/ArrowLeft.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTLF = visual.ImageStim(
        win=win,
        name='image_Cross_HTLF', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTLF = visual.ImageStim(
        win=win,
        name='image_Tar_HTLF', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTLF = keyboard.Keyboard(deviceName='key_HTLF')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "HTLU" ---
    image1_HTLU = visual.ImageStim(
        win=win,
        name='image1_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTLU = visual.ImageStim(
        win=win,
        name='image2_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTLU = visual.ImageStim(
        win=win,
        name='image3_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTLU = visual.ImageStim(
        win=win,
        name='image4_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTLU = visual.ImageStim(
        win=win,
        name='image5_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTLU = visual.ImageStim(
        win=win,
        name='image6_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTLU = visual.ImageStim(
        win=win,
        name='image7_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTLU = visual.ImageStim(
        win=win,
        name='image8_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTLU = visual.ImageStim(
        win=win,
        name='image9_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTLU = visual.ImageStim(
        win=win,
        name='image10_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTLU = visual.ImageStim(
        win=win,
        name='image11_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTLU = visual.ImageStim(
        win=win,
        name='image12_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTLU = visual.ImageStim(
        win=win,
        name='image13_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTLU = visual.ImageStim(
        win=win,
        name='image14_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTLU = visual.ImageStim(
        win=win,
        name='image15_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTLU = visual.ImageStim(
        win=win,
        name='image16_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTLU = visual.ImageStim(
        win=win,
        name='image17_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTLU = visual.ImageStim(
        win=win,
        name='image18_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTLU = visual.ImageStim(
        win=win,
        name='image19_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTLU = visual.ImageStim(
        win=win,
        name='image20_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTLU = visual.ImageStim(
        win=win,
        name='image21_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTLU = visual.ImageStim(
        win=win,
        name='image22_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTLU = visual.ImageStim(
        win=win,
        name='image23_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTLU = visual.ImageStim(
        win=win,
        name='image24_HTLU', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTLU = visual.ImageStim(
        win=win,
        name='image_Frame_HTLU', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTLU = visual.ImageStim(
        win=win,
        name='image_Cue_HTLU', 
        image='images/ArrowRight.png', mask=None, anchor='center',
        ori=0.0, pos=(0.002, 0.0015), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTLU = visual.ImageStim(
        win=win,
        name='image_Cross_HTLU', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTLU = visual.ImageStim(
        win=win,
        name='image_Tar_HTLU', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTLU = keyboard.Keyboard(deviceName='key_HTLU')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "HTRF" ---
    image1_HTRF = visual.ImageStim(
        win=win,
        name='image1_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTRF = visual.ImageStim(
        win=win,
        name='image2_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTRF = visual.ImageStim(
        win=win,
        name='image3_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTRF = visual.ImageStim(
        win=win,
        name='image4_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTRF = visual.ImageStim(
        win=win,
        name='image5_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTRF = visual.ImageStim(
        win=win,
        name='image6_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTRF = visual.ImageStim(
        win=win,
        name='image7_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTRF = visual.ImageStim(
        win=win,
        name='image8_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTRF = visual.ImageStim(
        win=win,
        name='image9_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTRF = visual.ImageStim(
        win=win,
        name='image10_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTRF = visual.ImageStim(
        win=win,
        name='image11_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTRF = visual.ImageStim(
        win=win,
        name='image12_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTRF = visual.ImageStim(
        win=win,
        name='image13_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTRF = visual.ImageStim(
        win=win,
        name='image14_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTRF = visual.ImageStim(
        win=win,
        name='image15_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTRF = visual.ImageStim(
        win=win,
        name='image16_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTRF = visual.ImageStim(
        win=win,
        name='image17_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTRF = visual.ImageStim(
        win=win,
        name='image18_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTRF = visual.ImageStim(
        win=win,
        name='image19_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTRF = visual.ImageStim(
        win=win,
        name='image20_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTRF = visual.ImageStim(
        win=win,
        name='image21_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTRF = visual.ImageStim(
        win=win,
        name='image22_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTRF = visual.ImageStim(
        win=win,
        name='image23_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTRF = visual.ImageStim(
        win=win,
        name='image24_HTRF', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTRF = visual.ImageStim(
        win=win,
        name='image_Frame_HTRF', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTRF = visual.ImageStim(
        win=win,
        name='image_Cue_HTRF', 
        image='images/ArrowRight.png', mask=None, anchor='center',
        ori=0.0, pos=(0.002, 0.0015), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTRF = visual.ImageStim(
        win=win,
        name='image_Cross_HTRF', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTRF = visual.ImageStim(
        win=win,
        name='image_Tar_HTRF', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTRF = keyboard.Keyboard(deviceName='key_HTRF')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "HTRU" ---
    image1_HTRU = visual.ImageStim(
        win=win,
        name='image1_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTRU = visual.ImageStim(
        win=win,
        name='image2_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTRU = visual.ImageStim(
        win=win,
        name='image3_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTRU = visual.ImageStim(
        win=win,
        name='image4_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTRU = visual.ImageStim(
        win=win,
        name='image5_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTRU = visual.ImageStim(
        win=win,
        name='image6_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTRU = visual.ImageStim(
        win=win,
        name='image7_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTRU = visual.ImageStim(
        win=win,
        name='image8_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTRU = visual.ImageStim(
        win=win,
        name='image9_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTRU = visual.ImageStim(
        win=win,
        name='image10_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTRU = visual.ImageStim(
        win=win,
        name='image11_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTRU = visual.ImageStim(
        win=win,
        name='image12_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTRU = visual.ImageStim(
        win=win,
        name='image13_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTRU = visual.ImageStim(
        win=win,
        name='image14_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTRU = visual.ImageStim(
        win=win,
        name='image15_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTRU = visual.ImageStim(
        win=win,
        name='image16_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTRU = visual.ImageStim(
        win=win,
        name='image17_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTRU = visual.ImageStim(
        win=win,
        name='image18_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTRU = visual.ImageStim(
        win=win,
        name='image19_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTRU = visual.ImageStim(
        win=win,
        name='image20_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTRU = visual.ImageStim(
        win=win,
        name='image21_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTRU = visual.ImageStim(
        win=win,
        name='image22_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTRU = visual.ImageStim(
        win=win,
        name='image23_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTRU = visual.ImageStim(
        win=win,
        name='image24_HTRU', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTRU = visual.ImageStim(
        win=win,
        name='image_Frame_HTRU', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTRU = visual.ImageStim(
        win=win,
        name='image_Cue_HTRU', 
        image='images/ArrowLeft.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTRU = visual.ImageStim(
        win=win,
        name='image_Cross_HTRU', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTRU = visual.ImageStim(
        win=win,
        name='image_Tar_HTRU', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTRU = keyboard.Keyboard(deviceName='key_HTRU')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "HTLF_OD" ---
    image1_HTLF_OD = visual.ImageStim(
        win=win,
        name='image1_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTLF_OD = visual.ImageStim(
        win=win,
        name='image2_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTLF_OD = visual.ImageStim(
        win=win,
        name='image3_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTLF_OD = visual.ImageStim(
        win=win,
        name='image4_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTLF_OD = visual.ImageStim(
        win=win,
        name='image5_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTLF_OD = visual.ImageStim(
        win=win,
        name='image6_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTLF_OD = visual.ImageStim(
        win=win,
        name='image7_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTLF_OD = visual.ImageStim(
        win=win,
        name='image8_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTLF_OD = visual.ImageStim(
        win=win,
        name='image9_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTLF_OD = visual.ImageStim(
        win=win,
        name='image10_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTLF_OD = visual.ImageStim(
        win=win,
        name='image11_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTLF_OD = visual.ImageStim(
        win=win,
        name='image12_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTLF_OD = visual.ImageStim(
        win=win,
        name='image13_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTLF_OD = visual.ImageStim(
        win=win,
        name='image14_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTLF_OD = visual.ImageStim(
        win=win,
        name='image15_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTLF_OD = visual.ImageStim(
        win=win,
        name='image16_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTLF_OD = visual.ImageStim(
        win=win,
        name='image17_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTLF_OD = visual.ImageStim(
        win=win,
        name='image18_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTLF_OD = visual.ImageStim(
        win=win,
        name='image19_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTLF_OD = visual.ImageStim(
        win=win,
        name='image20_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTLF_OD = visual.ImageStim(
        win=win,
        name='image21_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTLF_OD = visual.ImageStim(
        win=win,
        name='image22_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTLF_OD = visual.ImageStim(
        win=win,
        name='image23_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTLF_OD = visual.ImageStim(
        win=win,
        name='image24_HTLF_OD', units='norm', 
        image='images/pattern_images_rec_half_left/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTLF_OD = visual.ImageStim(
        win=win,
        name='image_Frame_HTLF_OD', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTLF_OD = visual.ImageStim(
        win=win,
        name='image_Cue_HTLF_OD', 
        image='images/ArrowLeft.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTLF_OD = visual.ImageStim(
        win=win,
        name='image_Cross_HTLF_OD', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTLF_OD = visual.ImageStim(
        win=win,
        name='image_Tar_HTLF_OD', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTLF_OD = keyboard.Keyboard(deviceName='key_HTLF_OD')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "HTRF_OD" ---
    image1_HTRF_OD = visual.ImageStim(
        win=win,
        name='image1_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_001.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image2_HTRF_OD = visual.ImageStim(
        win=win,
        name='image2_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_002.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image3_HTRF_OD = visual.ImageStim(
        win=win,
        name='image3_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_003.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image4_HTRF_OD = visual.ImageStim(
        win=win,
        name='image4_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_004.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image5_HTRF_OD = visual.ImageStim(
        win=win,
        name='image5_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_005.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image6_HTRF_OD = visual.ImageStim(
        win=win,
        name='image6_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_006.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    image7_HTRF_OD = visual.ImageStim(
        win=win,
        name='image7_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_007.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    image8_HTRF_OD = visual.ImageStim(
        win=win,
        name='image8_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_008.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    image9_HTRF_OD = visual.ImageStim(
        win=win,
        name='image9_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_009.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    image10_HTRF_OD = visual.ImageStim(
        win=win,
        name='image10_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_010.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    image11_HTRF_OD = visual.ImageStim(
        win=win,
        name='image11_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_011.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    image12_HTRF_OD = visual.ImageStim(
        win=win,
        name='image12_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_012.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    image13_HTRF_OD = visual.ImageStim(
        win=win,
        name='image13_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_013.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    image14_HTRF_OD = visual.ImageStim(
        win=win,
        name='image14_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_014.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-13.0)
    image15_HTRF_OD = visual.ImageStim(
        win=win,
        name='image15_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_015.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-14.0)
    image16_HTRF_OD = visual.ImageStim(
        win=win,
        name='image16_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_016.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-15.0)
    image17_HTRF_OD = visual.ImageStim(
        win=win,
        name='image17_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_017.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-16.0)
    image18_HTRF_OD = visual.ImageStim(
        win=win,
        name='image18_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_018.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-17.0)
    image19_HTRF_OD = visual.ImageStim(
        win=win,
        name='image19_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_019.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-18.0)
    image20_HTRF_OD = visual.ImageStim(
        win=win,
        name='image20_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_020.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-19.0)
    image21_HTRF_OD = visual.ImageStim(
        win=win,
        name='image21_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_021.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-20.0)
    image22_HTRF_OD = visual.ImageStim(
        win=win,
        name='image22_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_022.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-21.0)
    image23_HTRF_OD = visual.ImageStim(
        win=win,
        name='image23_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_023.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-22.0)
    image24_HTRF_OD = visual.ImageStim(
        win=win,
        name='image24_HTRF_OD', units='norm', 
        image='images/pattern_images_rec_half_right/pattern_024.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-23.0)
    image_Frame_HTRF_OD = visual.ImageStim(
        win=win,
        name='image_Frame_HTRF_OD', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-24.0)
    image_Cue_HTRF_OD = visual.ImageStim(
        win=win,
        name='image_Cue_HTRF_OD', 
        image='images/ArrowRight.png', mask=None, anchor='center',
        ori=0.0, pos=(0.002, 0.0015), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-25.0)
    image_Cross_HTRF_OD = visual.ImageStim(
        win=win,
        name='image_Cross_HTRF_OD', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-26.0)
    image_Tar_HTRF_OD = visual.ImageStim(
        win=win,
        name='image_Tar_HTRF_OD', 
        image='images/Target.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-27.0)
    key_HTRF_OD = keyboard.Keyboard(deviceName='key_HTRF_OD')
    
    # --- Initialize components for Routine "Break" ---
    image_Frame_Break = visual.ImageStim(
        win=win,
        name='image_Frame_Break', units='norm', 
        image='images/light_blue_frame_overlay.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_Cross_Break = visual.ImageStim(
        win=win,
        name='image_Cross_Break', 
        image='images/Cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- CUSTOM: Start  --- 
    my_numtot = 9 # total number of stimuli in one block
    my_numval = 6 # number of valid stimuli in one block
    my_numinval = my_numtot - my_numval # number of invalid stimuli in one block
    my_outerloopnreps = 31 # number of repetitions of outerloop
    my_numcond = 6 # total number of conditions in paradigm
    my_totblocktime = 1.6
    
    print(expInfo['participant'])
    my_seq = open('sequences_6c/CounterbalancedSequence_6c_'+str(expInfo['participant'])+'.txt', "r")
    my_seq_txt = my_seq.read()
    my_seq_num = np.zeros((my_outerloopnreps, 1))
    my_randtartime = np.random.randint(11,16,(my_numtot*my_outerloopnreps, 1))/10
    
    for i in range(my_outerloopnreps):
        my_seq_num[i] = int(my_seq_txt[2*i])
    
    my_rankTDA_L = np.zeros((my_numval, my_outerloopnreps))     # define array for ranks of valid stimuli in condition L
    my_rankTDA_R = np.zeros((my_numval, my_outerloopnreps))     # define array for ranks of valid stimuli in condition R
    my_rankTDA_GL = np.zeros((my_numval, my_outerloopnreps))    # define array for ranks of valid stimuli in condition GL
    my_rankTDA_GR = np.zeros((my_numval, my_outerloopnreps))    # define array for ranks of valid stimuli in condition GR
    my_rankTDA_L_OD = np.zeros((my_numval, my_outerloopnreps))  # define array for ranks of valid stimuli in condition L_OD
    my_rankTDA_R_OD = np.zeros((my_numval, my_outerloopnreps))  # define array for ranks of valid stimuli in condition R_OD
    
    my_corrResp_L = []     # define empty list for correct responses
    my_corrResp_R = []     # define empty list for correct responses
    my_corrResp_GL = []    # define empty list for correct responses
    my_corrResp_GR = []    # define empty list for correct responses
    my_corrResp_L_OD = []  # define empty list for correct responses 
    my_corrResp_R_OD = []  # define empty list for correct responses 
    
    for i in range(my_outerloopnreps): # loop over outerloops and choose ranks for stimuli in each iteration
        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_L[0:my_numval, i] = my_temp[0:my_numval] # choose ranks for valid stimuli 
        #my_rankTDA_L[0, i] = np.random.randint(low=0, high=10, size=None, dtype=int)
        #my_rankTDA_L[1, i] = np.random.randint(low=0, high=10, size=None, dtype=int)
        #while my_rankTDA_L[0, i] == my_rankTDA_L[1, i]:
            #my_rankTDA_L[1, i] = np.random.randint(low=0, high=8, size=None, dtype=int)

        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_R[0:my_numval, i] = my_temp[0:my_numval] # choose ranks for valid stimuli 
        #my_rankTDA_R[0, i] = np.random.randint(low=0, high=8, size=None, dtype=int)
        #my_rankTDA_R[1, i] = np.random.randint(low=0, high=8, size=None, dtype=int)
        #while my_rankTDA_R[0, i] == my_rankTDA_R[1, i]:
            #my_rankTDA_R[1, i] = np.random.randint(low=0, high=8, size=None, dtype=int)
        
        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_GL[0:my_numval, i] = my_temp[0:my_numval] # choose ranks for valid stimuli 
        #my_rankTDA_GL[0, i] = my_temp[0]
        #my_rankTDA_GL[1, i] = my_temp[1]
        #my_rankTDA_GL[2, i] = my_temp[2]
        #my_rankTDA_GL[3, i] = my_temp[3]
        #my_rankTDA_GL[4, i] = my_temp[4]
        
        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_GR[0:my_numval, i] = my_temp[0:my_numval] # choose ranks for valid stimuli 
        #my_rankTDA_GR[1, i] = my_temp[1]
        #my_rankTDA_GR[2, i] = my_temp[2]
        #my_rankTDA_GR[3, i] = my_temp[3]
        #my_rankTDA_GR[4, i] = my_temp[4]
        
        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_L_OD[0:my_numval, i] = my_temp[0:my_numval] 
        
        my_temp = np.random.permutation(my_numtot) # random permutation of numbers 0 to 9
        my_rankTDA_R_OD[0:my_numval, i] = my_temp[0:my_numval] 
    
    my_coordTDA_L = np.zeros((2, my_numtot, my_outerloopnreps))     # define array for coordinates of stimuli (1. Dimension: x,y; 2. Dim: stimuli of one block; 3. Dim: repetitions outerloop)
    my_coordTDA_R = np.zeros((2, my_numtot, my_outerloopnreps))     # define array for coordinates of stimuli
    my_coordTDA_GL = np.zeros((2, my_numtot, my_outerloopnreps))    # define array for coordinates of stimuli
    my_coordTDA_GR = np.zeros((2, my_numtot, my_outerloopnreps))    # define array for coordinates of stimuli
    my_coordTDA_L_OD = np.zeros((2, my_numtot, my_outerloopnreps))  # define array for coordinates of stimuli 
    my_coordTDA_R_OD = np.zeros((2, my_numtot, my_outerloopnreps))  # define array for coordinates of stimuli 
        
    for i in range(my_outerloopnreps): # loop over outerloops and fill arrays for coordinates and lists for correct responses
        my_corrResp_L.append([]) 
        my_corrResp_R.append([])
        my_corrResp_GL.append([])
        my_corrResp_GR.append([])
        my_corrResp_L_OD.append([])
        my_corrResp_R_OD.append([])
        
        for j in range(my_numtot):
            if (j in my_rankTDA_L[0:, i]):
                my_coordTDA_L[0,j,i] = np.random.rand()*0.2-0.4
                my_corrResp_L[i].append(1)
            else: 
                my_coordTDA_L[0,j,i] = np.random.rand()*0.2+0.2
                my_corrResp_L[i].append(None)
            my_coordTDA_L[1, j, i] = np.random.rand()*0.8-0.4
            
            
            #if (j == my_rankTDA_L[0, i]) or (j == my_rankTDA_L[1, i]):
            #    my_coordTDA_L[0, j, i] = np.random.rand()*0.2-0.4
            #    my_corrResp_L[i].append(1)
            #else:
            #   my_coordTDA_L[0, j, i] = np.random.rand()*0.2+0.2
            #    my_corrResp_L[i].append(None)
            #my_coordTDA_L[1, j, i] = np.random.rand()*0.8-0.4

        
            if (j in my_rankTDA_R[0:, i]):
                my_coordTDA_R[0, j, i] = np.random.rand()*0.2+0.2
                my_corrResp_R[i].append(1)
            else:
                my_coordTDA_R[0, j, i] = np.random.rand()*0.2-0.4
                my_corrResp_R[i].append(None)
            my_coordTDA_R[1, j, i] = np.random.rand()*0.8-0.4
            
            
            if (j in my_rankTDA_GL[0:, i]):
                my_coordTDA_GL[0, j, i] = np.random.rand()*0.2+0.2
                my_corrResp_GL[i].append(1)
            else:
                my_coordTDA_GL[0, j, i] = np.random.rand()*0.2-0.4
                my_corrResp_GL[i].append(None)
            my_coordTDA_GL[1, j, i] = np.random.rand()*0.8-0.4
            
            if (j in my_rankTDA_GR[0:, i]):
                my_coordTDA_GR[0, j, i] = np.random.rand()*0.2-0.4
                my_corrResp_GR[i].append(1)
            else:
                my_coordTDA_GR[0, j, i] = np.random.rand()*0.2+0.2
                my_corrResp_GR[i].append(None)
            my_coordTDA_GR[1, j, i] = np.random.rand()*0.8-0.4
            
            if (j in my_rankTDA_L_OD[0:, i]):
                my_coordTDA_L_OD[0, j, i] = np.random.rand()*0.2-0.4 
                my_corrResp_L_OD[i].append(1)
            else: 
                my_coordTDA_L_OD[0, j, i] = 2.0
                my_corrResp_L_OD[i].append(None)
            my_coordTDA_L_OD[1, j, i] = np.random.rand()*0.8-0.4 
            
            if (j in my_rankTDA_R_OD[0:, i]):
                my_coordTDA_R_OD[0, j, i] = np.random.rand()*0.2+0.2 
                my_corrResp_R_OD[i].append(1)
            else: 
                my_coordTDA_R_OD[0, j, i] = 2.0
                my_corrResp_R_OD[i].append(None)
            my_coordTDA_R_OD[1, j, i] = np.random.rand()*0.8-0.4 
            
    print("my_coordTDA_L")
    print(my_coordTDA_L)
    print("")
    print("my_coordTDA_R")
    print(my_coordTDA_R)
    print("")
    print("my_coordTDA_GL")
    print(my_coordTDA_GL)
    print("")
    print("my_coordTDA_GR")
    print(my_coordTDA_GR)
    print("")
    
    print("my_corrResp_L")
    print(my_corrResp_L)
    print("")
    print("my_corrResp_R")
    print(my_corrResp_R)
    print("")
    print("my_corrResp_GL")
    print(my_corrResp_GL)
    print("")
    print("my_corrResp_GR")
    print(my_corrResp_GR)
    print("")
    
    # --- CUSTOM: End  --- 
    
    
    # --- Prepare to start Routine "Trigger" ---
    # create an object to store info about Routine Trigger
    Trigger = data.Routine(
        name='Trigger',
        components=[],
    )
    Trigger.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Trigger
    Trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Trigger.tStart = globalClock.getTime(format='float')
    Trigger.status = STARTED
    thisExp.addData('Trigger.started', Trigger.tStart)
    Trigger.maxDuration = None
    # keep track of which components have finished
    TriggerComponents = Trigger.components
    for thisComponent in Trigger.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Trigger" ---
    Trigger.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Trigger.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Trigger.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Trigger" ---
    for thisComponent in Trigger.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Trigger
    Trigger.tStop = globalClock.getTime(format='float')
    Trigger.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Trigger.stopped', Trigger.tStop)
    thisExp.nextEntry()
    # the Routine "Trigger" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Outerloop = data.TrialHandler2(
        name='Outerloop',
        nReps=31.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(Outerloop)  # add the loop to the experiment
    thisOuterloop = Outerloop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOuterloop.rgb)
    if thisOuterloop != None:
        for paramName in thisOuterloop:
            globals()[paramName] = thisOuterloop[paramName]
    
    
    #--- CUSTOM START ---
    my_randtartime_counter = 0
    my_outerloopcounter = 0
    #--- CUSTOM END ---
    
    for thisOuterloop in Outerloop:
        currentLoop = Outerloop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisOuterloop.rgb)
        if thisOuterloop != None:
            for paramName in thisOuterloop:
                globals()[paramName] = thisOuterloop[paramName]
        
        if my_seq_num[my_outerloopcounter]==1:
        #Custom-Comment: Start TDA_L
        
            # set up handler to look after randomisation of conditions etc
            Trials_HTLF = data.TrialHandler2(
                name='Trials_HTLF',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TL.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(Trials_HTLF)  # add the loop to the experiment
            thisTrials_HTLF = Trials_HTLF.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTLF.rgb)
            if thisTrials_HTLF != None:
                for paramName in thisTrials_HTLF:
                    globals()[paramName] = thisTrials_HTLF[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDAL_counter = 0
            #--- CUSTOM END ---
            
            for thisTrials_HTLF in Trials_HTLF:
                currentLoop = Trials_HTLF
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTLF.rgb)
                if thisTrials_HTLF != None:
                    for paramName in thisTrials_HTLF:
                        globals()[paramName] = thisTrials_HTLF[paramName]
                
                # --- Prepare to start Routine "HTLF" ---
                # create an object to store info about Routine HTLF
                HTLF = data.Routine(
                    name='HTLF',
                    components=[image1_HTLF, image2_HTLF, image3_HTLF, image4_HTLF, image5_HTLF, image6_HTLF, image7_HTLF, image8_HTLF, image9_HTLF, image10_HTLF, image11_HTLF, image12_HTLF, image13_HTLF, image14_HTLF, image15_HTLF, image16_HTLF, image17_HTLF, image18_HTLF, image19_HTLF, image20_HTLF, image21_HTLF, image22_HTLF, image23_HTLF, image24_HTLF, image_Frame_HTLF, image_Cue_HTLF, image_Cross_HTLF, image_Tar_HTLF, key_HTLF],
                )
                HTLF.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                #--- CUSTOM START ---
                image_Tar_HTLF.setPos((my_coordTDA_L[0, my_TDAL_counter, my_outerloopcounter], my_coordTDA_L[1, my_TDAL_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTLF.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTLF
                key_HTLF.keys = []
                key_HTLF.rt = []
                _key_HTLF_allKeys = []
                # store start times for HTLF
                HTLF.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTLF.tStart = globalClock.getTime(format='float')
                HTLF.status = STARTED
                thisExp.addData('HTLF.started', HTLF.tStart)
                HTLF.maxDuration = None
                # keep track of which components have finished
                HTLFComponents = HTLF.components
                for thisComponent in HTLF.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTLF" ---
                # if trial has changed, end Routine now
                if isinstance(Trials_HTLF, data.TrialHandler2) and thisTrials_HTLF.thisN != Trials_HTLF.thisTrial.thisN:
                    continueRoutine = False
                HTLF.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTLF* updates
                    
                    # if image1_HTLF is starting this frame...
                    if image1_HTLF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTLF.frameNStart = frameN  # exact frame index
                        image1_HTLF.tStart = t  # local t and not account for scr refresh
                        image1_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTLF.started')
                        # update status
                        image1_HTLF.status = STARTED
                        image1_HTLF.setAutoDraw(True)
                    
                    # if image1_HTLF is active this frame...
                    if image1_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTLF is stopping this frame...
                    if image1_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTLF.tStop = t  # not accounting for scr refresh
                            image1_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTLF.stopped')
                            # update status
                            image1_HTLF.status = FINISHED
                            image1_HTLF.setAutoDraw(False)
                    
                    # *image2_HTLF* updates
                    
                    # if image2_HTLF is starting this frame...
                    if image2_HTLF.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTLF.frameNStart = frameN  # exact frame index
                        image2_HTLF.tStart = t  # local t and not account for scr refresh
                        image2_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTLF.started')
                        # update status
                        image2_HTLF.status = STARTED
                        image2_HTLF.setAutoDraw(True)
                    
                    # if image2_HTLF is active this frame...
                    if image2_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTLF is stopping this frame...
                    if image2_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTLF.tStop = t  # not accounting for scr refresh
                            image2_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTLF.stopped')
                            # update status
                            image2_HTLF.status = FINISHED
                            image2_HTLF.setAutoDraw(False)
                    
                    # *image3_HTLF* updates
                    
                    # if image3_HTLF is starting this frame...
                    if image3_HTLF.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTLF.frameNStart = frameN  # exact frame index
                        image3_HTLF.tStart = t  # local t and not account for scr refresh
                        image3_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTLF.started')
                        # update status
                        image3_HTLF.status = STARTED
                        image3_HTLF.setAutoDraw(True)
                    
                    # if image3_HTLF is active this frame...
                    if image3_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTLF is stopping this frame...
                    if image3_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTLF.tStop = t  # not accounting for scr refresh
                            image3_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTLF.stopped')
                            # update status
                            image3_HTLF.status = FINISHED
                            image3_HTLF.setAutoDraw(False)
                    
                    # *image4_HTLF* updates
                    
                    # if image4_HTLF is starting this frame...
                    if image4_HTLF.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTLF.frameNStart = frameN  # exact frame index
                        image4_HTLF.tStart = t  # local t and not account for scr refresh
                        image4_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTLF.started')
                        # update status
                        image4_HTLF.status = STARTED
                        image4_HTLF.setAutoDraw(True)
                    
                    # if image4_HTLF is active this frame...
                    if image4_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTLF is stopping this frame...
                    if image4_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTLF.tStop = t  # not accounting for scr refresh
                            image4_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTLF.stopped')
                            # update status
                            image4_HTLF.status = FINISHED
                            image4_HTLF.setAutoDraw(False)
                    
                    # *image5_HTLF* updates
                    
                    # if image5_HTLF is starting this frame...
                    if image5_HTLF.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTLF.frameNStart = frameN  # exact frame index
                        image5_HTLF.tStart = t  # local t and not account for scr refresh
                        image5_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTLF.started')
                        # update status
                        image5_HTLF.status = STARTED
                        image5_HTLF.setAutoDraw(True)
                    
                    # if image5_HTLF is active this frame...
                    if image5_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTLF is stopping this frame...
                    if image5_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTLF.tStop = t  # not accounting for scr refresh
                            image5_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTLF.stopped')
                            # update status
                            image5_HTLF.status = FINISHED
                            image5_HTLF.setAutoDraw(False)
                    
                    # *image6_HTLF* updates
                    
                    # if image6_HTLF is starting this frame...
                    if image6_HTLF.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTLF.frameNStart = frameN  # exact frame index
                        image6_HTLF.tStart = t  # local t and not account for scr refresh
                        image6_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTLF.started')
                        # update status
                        image6_HTLF.status = STARTED
                        image6_HTLF.setAutoDraw(True)
                    
                    # if image6_HTLF is active this frame...
                    if image6_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTLF is stopping this frame...
                    if image6_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTLF.tStop = t  # not accounting for scr refresh
                            image6_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTLF.stopped')
                            # update status
                            image6_HTLF.status = FINISHED
                            image6_HTLF.setAutoDraw(False)
                    
                    # *image7_HTLF* updates
                    
                    # if image7_HTLF is starting this frame...
                    if image7_HTLF.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTLF.frameNStart = frameN  # exact frame index
                        image7_HTLF.tStart = t  # local t and not account for scr refresh
                        image7_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTLF.started')
                        # update status
                        image7_HTLF.status = STARTED
                        image7_HTLF.setAutoDraw(True)
                    
                    # if image7_HTLF is active this frame...
                    if image7_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTLF is stopping this frame...
                    if image7_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTLF.tStop = t  # not accounting for scr refresh
                            image7_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTLF.stopped')
                            # update status
                            image7_HTLF.status = FINISHED
                            image7_HTLF.setAutoDraw(False)
                    
                    # *image8_HTLF* updates
                    
                    # if image8_HTLF is starting this frame...
                    if image8_HTLF.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTLF.frameNStart = frameN  # exact frame index
                        image8_HTLF.tStart = t  # local t and not account for scr refresh
                        image8_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTLF.started')
                        # update status
                        image8_HTLF.status = STARTED
                        image8_HTLF.setAutoDraw(True)
                    
                    # if image8_HTLF is active this frame...
                    if image8_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTLF is stopping this frame...
                    if image8_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTLF.tStop = t  # not accounting for scr refresh
                            image8_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTLF.stopped')
                            # update status
                            image8_HTLF.status = FINISHED
                            image8_HTLF.setAutoDraw(False)
                    
                    # *image9_HTLF* updates
                    
                    # if image9_HTLF is starting this frame...
                    if image9_HTLF.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTLF.frameNStart = frameN  # exact frame index
                        image9_HTLF.tStart = t  # local t and not account for scr refresh
                        image9_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTLF.started')
                        # update status
                        image9_HTLF.status = STARTED
                        image9_HTLF.setAutoDraw(True)
                    
                    # if image9_HTLF is active this frame...
                    if image9_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTLF is stopping this frame...
                    if image9_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTLF.tStop = t  # not accounting for scr refresh
                            image9_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTLF.stopped')
                            # update status
                            image9_HTLF.status = FINISHED
                            image9_HTLF.setAutoDraw(False)
                    
                    # *image10_HTLF* updates
                    
                    # if image10_HTLF is starting this frame...
                    if image10_HTLF.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTLF.frameNStart = frameN  # exact frame index
                        image10_HTLF.tStart = t  # local t and not account for scr refresh
                        image10_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTLF.started')
                        # update status
                        image10_HTLF.status = STARTED
                        image10_HTLF.setAutoDraw(True)
                    
                    # if image10_HTLF is active this frame...
                    if image10_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTLF is stopping this frame...
                    if image10_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTLF.tStop = t  # not accounting for scr refresh
                            image10_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTLF.stopped')
                            # update status
                            image10_HTLF.status = FINISHED
                            image10_HTLF.setAutoDraw(False)
                    
                    # *image11_HTLF* updates
                    
                    # if image11_HTLF is starting this frame...
                    if image11_HTLF.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTLF.frameNStart = frameN  # exact frame index
                        image11_HTLF.tStart = t  # local t and not account for scr refresh
                        image11_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTLF.started')
                        # update status
                        image11_HTLF.status = STARTED
                        image11_HTLF.setAutoDraw(True)
                    
                    # if image11_HTLF is active this frame...
                    if image11_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTLF is stopping this frame...
                    if image11_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTLF.tStop = t  # not accounting for scr refresh
                            image11_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTLF.stopped')
                            # update status
                            image11_HTLF.status = FINISHED
                            image11_HTLF.setAutoDraw(False)
                    
                    # *image12_HTLF* updates
                    
                    # if image12_HTLF is starting this frame...
                    if image12_HTLF.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTLF.frameNStart = frameN  # exact frame index
                        image12_HTLF.tStart = t  # local t and not account for scr refresh
                        image12_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTLF.started')
                        # update status
                        image12_HTLF.status = STARTED
                        image12_HTLF.setAutoDraw(True)
                    
                    # if image12_HTLF is active this frame...
                    if image12_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTLF is stopping this frame...
                    if image12_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTLF.tStop = t  # not accounting for scr refresh
                            image12_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTLF.stopped')
                            # update status
                            image12_HTLF.status = FINISHED
                            image12_HTLF.setAutoDraw(False)
                    
                    # *image13_HTLF* updates
                    
                    # if image13_HTLF is starting this frame...
                    if image13_HTLF.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTLF.frameNStart = frameN  # exact frame index
                        image13_HTLF.tStart = t  # local t and not account for scr refresh
                        image13_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTLF.started')
                        # update status
                        image13_HTLF.status = STARTED
                        image13_HTLF.setAutoDraw(True)
                    
                    # if image13_HTLF is active this frame...
                    if image13_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTLF is stopping this frame...
                    if image13_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTLF.tStop = t  # not accounting for scr refresh
                            image13_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTLF.stopped')
                            # update status
                            image13_HTLF.status = FINISHED
                            image13_HTLF.setAutoDraw(False)
                    
                    # *image14_HTLF* updates
                    
                    # if image14_HTLF is starting this frame...
                    if image14_HTLF.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTLF.frameNStart = frameN  # exact frame index
                        image14_HTLF.tStart = t  # local t and not account for scr refresh
                        image14_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTLF.started')
                        # update status
                        image14_HTLF.status = STARTED
                        image14_HTLF.setAutoDraw(True)
                    
                    # if image14_HTLF is active this frame...
                    if image14_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTLF is stopping this frame...
                    if image14_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTLF.tStop = t  # not accounting for scr refresh
                            image14_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTLF.stopped')
                            # update status
                            image14_HTLF.status = FINISHED
                            image14_HTLF.setAutoDraw(False)
                    
                    # *image15_HTLF* updates
                    
                    # if image15_HTLF is starting this frame...
                    if image15_HTLF.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTLF.frameNStart = frameN  # exact frame index
                        image15_HTLF.tStart = t  # local t and not account for scr refresh
                        image15_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTLF.started')
                        # update status
                        image15_HTLF.status = STARTED
                        image15_HTLF.setAutoDraw(True)
                    
                    # if image15_HTLF is active this frame...
                    if image15_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTLF is stopping this frame...
                    if image15_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTLF.tStop = t  # not accounting for scr refresh
                            image15_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTLF.stopped')
                            # update status
                            image15_HTLF.status = FINISHED
                            image15_HTLF.setAutoDraw(False)
                    
                    # *image16_HTLF* updates
                    
                    # if image16_HTLF is starting this frame...
                    if image16_HTLF.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTLF.frameNStart = frameN  # exact frame index
                        image16_HTLF.tStart = t  # local t and not account for scr refresh
                        image16_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTLF.started')
                        # update status
                        image16_HTLF.status = STARTED
                        image16_HTLF.setAutoDraw(True)
                    
                    # if image16_HTLF is active this frame...
                    if image16_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTLF is stopping this frame...
                    if image16_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTLF.tStop = t  # not accounting for scr refresh
                            image16_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTLF.stopped')
                            # update status
                            image16_HTLF.status = FINISHED
                            image16_HTLF.setAutoDraw(False)
                    
                    # *image17_HTLF* updates
                    
                    # if image17_HTLF is starting this frame...
                    if image17_HTLF.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTLF.frameNStart = frameN  # exact frame index
                        image17_HTLF.tStart = t  # local t and not account for scr refresh
                        image17_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTLF.started')
                        # update status
                        image17_HTLF.status = STARTED
                        image17_HTLF.setAutoDraw(True)
                    
                    # if image17_HTLF is active this frame...
                    if image17_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTLF is stopping this frame...
                    if image17_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTLF.tStop = t  # not accounting for scr refresh
                            image17_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTLF.stopped')
                            # update status
                            image17_HTLF.status = FINISHED
                            image17_HTLF.setAutoDraw(False)
                    
                    # *image18_HTLF* updates
                    
                    # if image18_HTLF is starting this frame...
                    if image18_HTLF.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTLF.frameNStart = frameN  # exact frame index
                        image18_HTLF.tStart = t  # local t and not account for scr refresh
                        image18_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTLF.started')
                        # update status
                        image18_HTLF.status = STARTED
                        image18_HTLF.setAutoDraw(True)
                    
                    # if image18_HTLF is active this frame...
                    if image18_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTLF is stopping this frame...
                    if image18_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTLF.tStop = t  # not accounting for scr refresh
                            image18_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTLF.stopped')
                            # update status
                            image18_HTLF.status = FINISHED
                            image18_HTLF.setAutoDraw(False)
                    
                    # *image19_HTLF* updates
                    
                    # if image19_HTLF is starting this frame...
                    if image19_HTLF.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTLF.frameNStart = frameN  # exact frame index
                        image19_HTLF.tStart = t  # local t and not account for scr refresh
                        image19_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTLF.started')
                        # update status
                        image19_HTLF.status = STARTED
                        image19_HTLF.setAutoDraw(True)
                    
                    # if image19_HTLF is active this frame...
                    if image19_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTLF is stopping this frame...
                    if image19_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTLF.tStop = t  # not accounting for scr refresh
                            image19_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTLF.stopped')
                            # update status
                            image19_HTLF.status = FINISHED
                            image19_HTLF.setAutoDraw(False)
                    
                    # *image20_HTLF* updates
                    
                    # if image20_HTLF is starting this frame...
                    if image20_HTLF.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTLF.frameNStart = frameN  # exact frame index
                        image20_HTLF.tStart = t  # local t and not account for scr refresh
                        image20_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTLF.started')
                        # update status
                        image20_HTLF.status = STARTED
                        image20_HTLF.setAutoDraw(True)
                    
                    # if image20_HTLF is active this frame...
                    if image20_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTLF is stopping this frame...
                    if image20_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTLF.tStop = t  # not accounting for scr refresh
                            image20_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTLF.stopped')
                            # update status
                            image20_HTLF.status = FINISHED
                            image20_HTLF.setAutoDraw(False)
                    
                    # *image21_HTLF* updates
                    
                    # if image21_HTLF is starting this frame...
                    if image21_HTLF.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTLF.frameNStart = frameN  # exact frame index
                        image21_HTLF.tStart = t  # local t and not account for scr refresh
                        image21_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTLF.started')
                        # update status
                        image21_HTLF.status = STARTED
                        image21_HTLF.setAutoDraw(True)
                    
                    # if image21_HTLF is active this frame...
                    if image21_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTLF is stopping this frame...
                    if image21_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTLF.tStop = t  # not accounting for scr refresh
                            image21_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTLF.stopped')
                            # update status
                            image21_HTLF.status = FINISHED
                            image21_HTLF.setAutoDraw(False)
                    
                    # *image22_HTLF* updates
                    
                    # if image22_HTLF is starting this frame...
                    if image22_HTLF.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTLF.frameNStart = frameN  # exact frame index
                        image22_HTLF.tStart = t  # local t and not account for scr refresh
                        image22_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTLF.started')
                        # update status
                        image22_HTLF.status = STARTED
                        image22_HTLF.setAutoDraw(True)
                    
                    # if image22_HTLF is active this frame...
                    if image22_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTLF is stopping this frame...
                    if image22_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTLF.tStop = t  # not accounting for scr refresh
                            image22_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTLF.stopped')
                            # update status
                            image22_HTLF.status = FINISHED
                            image22_HTLF.setAutoDraw(False)
                    
                    # *image23_HTLF* updates
                    
                    # if image23_HTLF is starting this frame...
                    if image23_HTLF.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTLF.frameNStart = frameN  # exact frame index
                        image23_HTLF.tStart = t  # local t and not account for scr refresh
                        image23_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTLF.started')
                        # update status
                        image23_HTLF.status = STARTED
                        image23_HTLF.setAutoDraw(True)
                    
                    # if image23_HTLF is active this frame...
                    if image23_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTLF is stopping this frame...
                    if image23_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTLF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTLF.tStop = t  # not accounting for scr refresh
                            image23_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTLF.stopped')
                            # update status
                            image23_HTLF.status = FINISHED
                            image23_HTLF.setAutoDraw(False)
                    
                    # *image24_HTLF* updates
                    
                    # if image24_HTLF is starting this frame...
                    if image24_HTLF.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTLF.frameNStart = frameN  # exact frame index
                        image24_HTLF.tStart = t  # local t and not account for scr refresh
                        image24_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTLF.started')
                        # update status
                        image24_HTLF.status = STARTED
                        image24_HTLF.setAutoDraw(True)
                    
                    # if image24_HTLF is active this frame...
                    if image24_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTLF is stopping this frame...
                    if image24_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTLF.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTLF.tStop = t  # not accounting for scr refresh
                            image24_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTLF.stopped')
                            # update status
                            image24_HTLF.status = FINISHED
                            image24_HTLF.setAutoDraw(False)
                    
                    # *image_Frame_HTLF* updates
                    
                    # if image_Frame_HTLF is starting this frame...
                    if image_Frame_HTLF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTLF.frameNStart = frameN  # exact frame index
                        image_Frame_HTLF.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTLF.started')
                        # update status
                        image_Frame_HTLF.status = STARTED
                        image_Frame_HTLF.setAutoDraw(True)
                    
                    # if image_Frame_HTLF is active this frame...
                    if image_Frame_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTLF is stopping this frame...
                    if image_Frame_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTLF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTLF.tStop = t  # not accounting for scr refresh
                            image_Frame_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTLF.stopped')
                            # update status
                            image_Frame_HTLF.status = FINISHED
                            image_Frame_HTLF.setAutoDraw(False)
                    
                    # *image_Cue_HTLF* updates
                    
                    # if image_Cue_HTLF is starting this frame...
                    if image_Cue_HTLF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTLF.frameNStart = frameN  # exact frame index
                        image_Cue_HTLF.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTLF.started')
                        # update status
                        image_Cue_HTLF.status = STARTED
                        image_Cue_HTLF.setAutoDraw(True)
                    
                    # if image_Cue_HTLF is active this frame...
                    if image_Cue_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTLF is stopping this frame...
                    if image_Cue_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTLF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTLF.tStop = t  # not accounting for scr refresh
                            image_Cue_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTLF.stopped')
                            # update status
                            image_Cue_HTLF.status = FINISHED
                            image_Cue_HTLF.setAutoDraw(False)
                    
                    # *image_Cross_HTLF* updates
                    
                    # if image_Cross_HTLF is starting this frame...
                    if image_Cross_HTLF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTLF.frameNStart = frameN  # exact frame index
                        image_Cross_HTLF.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTLF.started')
                        # update status
                        image_Cross_HTLF.status = STARTED
                        image_Cross_HTLF.setAutoDraw(True)
                    
                    # if image_Cross_HTLF is active this frame...
                    if image_Cross_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTLF is stopping this frame...
                    if image_Cross_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTLF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTLF.tStop = t  # not accounting for scr refresh
                            image_Cross_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTLF.stopped')
                            # update status
                            image_Cross_HTLF.status = FINISHED
                            image_Cross_HTLF.setAutoDraw(False)
                    
                    # *image_Tar_HTLF* updates
                    
                    # if image_Tar_HTLF is starting this frame...
                    if image_Tar_HTLF.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTLF.frameNStart = frameN  # exact frame index
                        image_Tar_HTLF.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTLF.started')
                        # update status
                        image_Tar_HTLF.status = STARTED
                        image_Tar_HTLF.setAutoDraw(True)
                    
                    # if image_Tar_HTLF is active this frame...
                    if image_Tar_HTLF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTLF is stopping this frame...
                    if image_Tar_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTLF.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END 
                            # keep track of stop time/frame for later
                            image_Tar_HTLF.tStop = t  # not accounting for scr refresh
                            image_Tar_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTLF.stopped')
                            # update status
                            image_Tar_HTLF.status = FINISHED
                            image_Tar_HTLF.setAutoDraw(False)
                    
                    # *key_HTLF* updates
                    waitOnFlip = False
                    
                    # if key_HTLF is starting this frame...
                    if key_HTLF.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTLF.frameNStart = frameN  # exact frame index
                        key_HTLF.tStart = t  # local t and not account for scr refresh
                        key_HTLF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTLF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTLF.started')
                        # update status
                        key_HTLF.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTLF.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTLF.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTLF is stopping this frame...
                    if key_HTLF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTLF.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTLF.tStop = t  # not accounting for scr refresh
                            key_HTLF.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTLF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTLF.stopped')
                            # update status
                            key_HTLF.status = FINISHED
                            key_HTLF.status = FINISHED
                    if key_HTLF.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTLF.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTLF_allKeys.extend(theseKeys)
                        if len(_key_HTLF_allKeys):
                            key_HTLF.keys = _key_HTLF_allKeys[0].name  # just the first key pressed
                            key_HTLF.rt = _key_HTLF_allKeys[0].rt
                            key_HTLF.duration = _key_HTLF_allKeys[0].duration
                            # was this correct?
                            #CUSTOM: Start
                            #if (key_HTLF.keys == str(corr_resp)) or (key_HTLF.keys == corr_resp):
                            if (key_HTLF.keys == str(my_corrResp_L[my_outerloopcounter][my_TDAL_counter])) or (key_HTLF.keys == my_corrResp_L[my_outerloopcounter][my_TDAL_counter]):
                                key_HTLF.corr = 1
                            else:
                                key_HTLF.corr = 0
                            #CUSTOM: End
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTLF.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTLF.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END
                
                # --- Ending Routine "HTLF" ---
                for thisComponent in HTLF.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTLF
                HTLF.tStop = globalClock.getTime(format='float')
                HTLF.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTLF.stopped', HTLF.tStop)
                # check responses
                if key_HTLF.keys in ['', [], None]:  # No response was made
                    key_HTLF.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_L[my_outerloopcounter][my_TDAL_counter]).lower() == 'none': ## CUSTOM START & END
                       key_HTLF.corr = 1;  # correct non-response
                    else:
                       key_HTLF.corr = 0;  # failed to respond (incorrectly)
                # store data for Trials_HTLF (TrialHandler)
                Trials_HTLF.addData('key_HTLF.keys',key_HTLF.keys)
                Trials_HTLF.addData('key_HTLF.corr', key_HTLF.corr)
                if key_HTLF.keys != None:  # we had a response
                    Trials_HTLF.addData('key_HTLF.rt', key_HTLF.rt)
                    Trials_HTLF.addData('key_HTLF.duration', key_HTLF.duration)
                try:
                    if image1_HTLF.tStopRefresh is not None:
                        duration_val = image1_HTLF.tStopRefresh - image1_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image1_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTLF).__name__,
                        trial_type='image1_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image1_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTLF.tStopRefresh is not None:
                        duration_val = image2_HTLF.tStopRefresh - image2_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image2_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTLF).__name__,
                        trial_type='image2_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image2_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTLF.tStopRefresh is not None:
                        duration_val = image3_HTLF.tStopRefresh - image3_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image3_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTLF).__name__,
                        trial_type='image3_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image3_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTLF.tStopRefresh is not None:
                        duration_val = image4_HTLF.tStopRefresh - image4_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image4_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTLF).__name__,
                        trial_type='image4_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image4_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTLF.tStopRefresh is not None:
                        duration_val = image5_HTLF.tStopRefresh - image5_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image5_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTLF).__name__,
                        trial_type='image5_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image5_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTLF.tStopRefresh is not None:
                        duration_val = image6_HTLF.tStopRefresh - image6_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image6_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTLF).__name__,
                        trial_type='image6_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image6_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTLF.tStopRefresh is not None:
                        duration_val = image7_HTLF.tStopRefresh - image7_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image7_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTLF).__name__,
                        trial_type='image7_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image7_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTLF.tStopRefresh is not None:
                        duration_val = image8_HTLF.tStopRefresh - image8_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image8_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTLF).__name__,
                        trial_type='image8_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image8_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTLF.tStopRefresh is not None:
                        duration_val = image9_HTLF.tStopRefresh - image9_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image9_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTLF).__name__,
                        trial_type='image9_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image9_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTLF.tStopRefresh is not None:
                        duration_val = image10_HTLF.tStopRefresh - image10_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image10_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTLF).__name__,
                        trial_type='image10_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image10_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTLF.tStopRefresh is not None:
                        duration_val = image11_HTLF.tStopRefresh - image11_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image11_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTLF).__name__,
                        trial_type='image11_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image11_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTLF.tStopRefresh is not None:
                        duration_val = image12_HTLF.tStopRefresh - image12_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image12_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTLF).__name__,
                        trial_type='image12_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image12_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTLF.tStopRefresh is not None:
                        duration_val = image13_HTLF.tStopRefresh - image13_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image13_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTLF).__name__,
                        trial_type='image13_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image13_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTLF.tStopRefresh is not None:
                        duration_val = image14_HTLF.tStopRefresh - image14_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image14_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTLF).__name__,
                        trial_type='image14_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image14_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTLF.tStopRefresh is not None:
                        duration_val = image15_HTLF.tStopRefresh - image15_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image15_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTLF).__name__,
                        trial_type='image15_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image15_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTLF.tStopRefresh is not None:
                        duration_val = image16_HTLF.tStopRefresh - image16_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image16_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTLF).__name__,
                        trial_type='image16_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image16_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTLF.tStopRefresh is not None:
                        duration_val = image17_HTLF.tStopRefresh - image17_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image17_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTLF).__name__,
                        trial_type='image17_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image17_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTLF.tStopRefresh is not None:
                        duration_val = image18_HTLF.tStopRefresh - image18_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image18_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTLF).__name__,
                        trial_type='image18_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image18_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTLF.tStopRefresh is not None:
                        duration_val = image19_HTLF.tStopRefresh - image19_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image19_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTLF).__name__,
                        trial_type='image19_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image19_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTLF.tStopRefresh is not None:
                        duration_val = image20_HTLF.tStopRefresh - image20_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image20_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTLF).__name__,
                        trial_type='image20_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image20_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTLF.tStopRefresh is not None:
                        duration_val = image21_HTLF.tStopRefresh - image21_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image21_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTLF).__name__,
                        trial_type='image21_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image21_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTLF.tStopRefresh is not None:
                        duration_val = image22_HTLF.tStopRefresh - image22_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image22_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTLF).__name__,
                        trial_type='image22_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image22_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTLF.tStopRefresh is not None:
                        duration_val = image23_HTLF.tStopRefresh - image23_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image23_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTLF).__name__,
                        trial_type='image23_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image23_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTLF.tStopRefresh is not None:
                        duration_val = image24_HTLF.tStopRefresh - image24_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image24_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTLF).__name__,
                        trial_type='image24_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image24_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTLF.tStopRefresh is not None:
                        duration_val = image_Cue_HTLF.tStopRefresh - image_Cue_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image_Cue_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTLF).__name__,
                        trial_type='image_Cue_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image_Cue_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTLF.tStopRefresh is not None:
                        duration_val = image_Tar_HTLF.tStopRefresh - image_Tar_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - image_Tar_HTLF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTLF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTLF).__name__,
                        trial_type='image_Tar_HTLF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_image_Tar_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTLF.tStopRefresh is not None:
                        duration_val = key_HTLF.tStopRefresh - key_HTLF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF.stopped'] - key_HTLF.tStartRefresh
                    if hasattr(key_HTLF, 'rt'):
                        rt_val = key_HTLF.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTLF" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTLF.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTLF).__name__,
                        trial_type='key_HTLF',
                        value=key_HTLF.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLF.addData('bidsEvent_key_HTLF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTLF.maxDurationReached:
                    routineTimer.addTime(-HTLF.maxDuration)
                elif HTLF.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                #--- CUSTOM START ---
                my_TDAL_counter = my_TDAL_counter + 1
                #--- CUSTOM END ---
                
            # completed 1.0 repeats of 'Trials_HTLF'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if Trials_HTLF.trialList in ([], [None], None):
                params = []
            else:
                params = Trials_HTLF.trialList[0].keys()
            # save data for this loop
            Trials_HTLF.saveAsExcel(filename + '.xlsx', sheetName='Trials_HTLF',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            Trials_HTLF.saveAsText(filename + 'Trials_HTLF.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
        
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
        #Custom-Comment: End TDA_L
        elif my_seq_num[my_outerloopcounter]==2:
        #Custom-Comment: Start TDA_R
            
            # set up handler to look after randomisation of conditions etc
            Trials_HTLU = data.TrialHandler2(
                name='Trials_HTLU',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TL.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(Trials_HTLU)  # add the loop to the experiment
            thisTrials_HTLU = Trials_HTLU.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTLU.rgb)
            if thisTrials_HTLU != None:
                for paramName in thisTrials_HTLU:
                    globals()[paramName] = thisTrials_HTLU[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDAR_counter = 0
            #--- CUSTOM END ---
            
            for thisTrials_HTLU in Trials_HTLU:
                currentLoop = Trials_HTLU
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTLU.rgb)
                if thisTrials_HTLU != None:
                    for paramName in thisTrials_HTLU:
                        globals()[paramName] = thisTrials_HTLU[paramName]
                
                # --- Prepare to start Routine "HTLU" ---
                # create an object to store info about Routine HTLU
                HTLU = data.Routine(
                    name='HTLU',
                    components=[image1_HTLU, image2_HTLU, image3_HTLU, image4_HTLU, image5_HTLU, image6_HTLU, image7_HTLU, image8_HTLU, image9_HTLU, image10_HTLU, image11_HTLU, image12_HTLU, image13_HTLU, image14_HTLU, image15_HTLU, image16_HTLU, image17_HTLU, image18_HTLU, image19_HTLU, image20_HTLU, image21_HTLU, image22_HTLU, image23_HTLU, image24_HTLU, image_Frame_HTLU, image_Cue_HTLU, image_Cross_HTLU, image_Tar_HTLU, key_HTLU],
                )
                HTLU.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                #--- CUSTOM START ---
                image_Tar_HTLU.setPos((my_coordTDA_R[0, my_TDAR_counter, my_outerloopcounter], my_coordTDA_R[1, my_TDAR_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTLU.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTLU
                key_HTLU.keys = []
                key_HTLU.rt = []
                _key_HTLU_allKeys = []
                # store start times for HTLU
                HTLU.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTLU.tStart = globalClock.getTime(format='float')
                HTLU.status = STARTED
                thisExp.addData('HTLU.started', HTLU.tStart)
                HTLU.maxDuration = None
                # keep track of which components have finished
                HTLUComponents = HTLU.components
                for thisComponent in HTLU.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTLU" ---
                # if trial has changed, end Routine now
                if isinstance(Trials_HTLU, data.TrialHandler2) and thisTrials_HTLU.thisN != Trials_HTLU.thisTrial.thisN:
                    continueRoutine = False
                HTLU.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTLU* updates
                    
                    # if image1_HTLU is starting this frame...
                    if image1_HTLU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTLU.frameNStart = frameN  # exact frame index
                        image1_HTLU.tStart = t  # local t and not account for scr refresh
                        image1_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTLU.started')
                        # update status
                        image1_HTLU.status = STARTED
                        image1_HTLU.setAutoDraw(True)
                    
                    # if image1_HTLU is active this frame...
                    if image1_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTLU is stopping this frame...
                    if image1_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTLU.tStop = t  # not accounting for scr refresh
                            image1_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTLU.stopped')
                            # update status
                            image1_HTLU.status = FINISHED
                            image1_HTLU.setAutoDraw(False)
                    
                    # *image2_HTLU* updates
                    
                    # if image2_HTLU is starting this frame...
                    if image2_HTLU.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTLU.frameNStart = frameN  # exact frame index
                        image2_HTLU.tStart = t  # local t and not account for scr refresh
                        image2_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTLU.started')
                        # update status
                        image2_HTLU.status = STARTED
                        image2_HTLU.setAutoDraw(True)
                    
                    # if image2_HTLU is active this frame...
                    if image2_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTLU is stopping this frame...
                    if image2_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTLU.tStop = t  # not accounting for scr refresh
                            image2_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTLU.stopped')
                            # update status
                            image2_HTLU.status = FINISHED
                            image2_HTLU.setAutoDraw(False)
                    
                    # *image3_HTLU* updates
                    
                    # if image3_HTLU is starting this frame...
                    if image3_HTLU.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTLU.frameNStart = frameN  # exact frame index
                        image3_HTLU.tStart = t  # local t and not account for scr refresh
                        image3_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTLU.started')
                        # update status
                        image3_HTLU.status = STARTED
                        image3_HTLU.setAutoDraw(True)
                    
                    # if image3_HTLU is active this frame...
                    if image3_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTLU is stopping this frame...
                    if image3_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTLU.tStop = t  # not accounting for scr refresh
                            image3_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTLU.stopped')
                            # update status
                            image3_HTLU.status = FINISHED
                            image3_HTLU.setAutoDraw(False)
                    
                    # *image4_HTLU* updates
                    
                    # if image4_HTLU is starting this frame...
                    if image4_HTLU.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTLU.frameNStart = frameN  # exact frame index
                        image4_HTLU.tStart = t  # local t and not account for scr refresh
                        image4_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTLU.started')
                        # update status
                        image4_HTLU.status = STARTED
                        image4_HTLU.setAutoDraw(True)
                    
                    # if image4_HTLU is active this frame...
                    if image4_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTLU is stopping this frame...
                    if image4_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTLU.tStop = t  # not accounting for scr refresh
                            image4_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTLU.stopped')
                            # update status
                            image4_HTLU.status = FINISHED
                            image4_HTLU.setAutoDraw(False)
                    
                    # *image5_HTLU* updates
                    
                    # if image5_HTLU is starting this frame...
                    if image5_HTLU.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTLU.frameNStart = frameN  # exact frame index
                        image5_HTLU.tStart = t  # local t and not account for scr refresh
                        image5_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTLU.started')
                        # update status
                        image5_HTLU.status = STARTED
                        image5_HTLU.setAutoDraw(True)
                    
                    # if image5_HTLU is active this frame...
                    if image5_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTLU is stopping this frame...
                    if image5_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTLU.tStop = t  # not accounting for scr refresh
                            image5_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTLU.stopped')
                            # update status
                            image5_HTLU.status = FINISHED
                            image5_HTLU.setAutoDraw(False)
                    
                    # *image6_HTLU* updates
                    
                    # if image6_HTLU is starting this frame...
                    if image6_HTLU.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTLU.frameNStart = frameN  # exact frame index
                        image6_HTLU.tStart = t  # local t and not account for scr refresh
                        image6_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTLU.started')
                        # update status
                        image6_HTLU.status = STARTED
                        image6_HTLU.setAutoDraw(True)
                    
                    # if image6_HTLU is active this frame...
                    if image6_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTLU is stopping this frame...
                    if image6_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTLU.tStop = t  # not accounting for scr refresh
                            image6_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTLU.stopped')
                            # update status
                            image6_HTLU.status = FINISHED
                            image6_HTLU.setAutoDraw(False)
                    
                    # *image7_HTLU* updates
                    
                    # if image7_HTLU is starting this frame...
                    if image7_HTLU.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTLU.frameNStart = frameN  # exact frame index
                        image7_HTLU.tStart = t  # local t and not account for scr refresh
                        image7_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTLU.started')
                        # update status
                        image7_HTLU.status = STARTED
                        image7_HTLU.setAutoDraw(True)
                    
                    # if image7_HTLU is active this frame...
                    if image7_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTLU is stopping this frame...
                    if image7_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTLU.tStop = t  # not accounting for scr refresh
                            image7_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTLU.stopped')
                            # update status
                            image7_HTLU.status = FINISHED
                            image7_HTLU.setAutoDraw(False)
                    
                    # *image8_HTLU* updates
                    
                    # if image8_HTLU is starting this frame...
                    if image8_HTLU.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTLU.frameNStart = frameN  # exact frame index
                        image8_HTLU.tStart = t  # local t and not account for scr refresh
                        image8_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTLU.started')
                        # update status
                        image8_HTLU.status = STARTED
                        image8_HTLU.setAutoDraw(True)
                    
                    # if image8_HTLU is active this frame...
                    if image8_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTLU is stopping this frame...
                    if image8_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTLU.tStop = t  # not accounting for scr refresh
                            image8_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTLU.stopped')
                            # update status
                            image8_HTLU.status = FINISHED
                            image8_HTLU.setAutoDraw(False)
                    
                    # *image9_HTLU* updates
                    
                    # if image9_HTLU is starting this frame...
                    if image9_HTLU.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTLU.frameNStart = frameN  # exact frame index
                        image9_HTLU.tStart = t  # local t and not account for scr refresh
                        image9_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTLU.started')
                        # update status
                        image9_HTLU.status = STARTED
                        image9_HTLU.setAutoDraw(True)
                    
                    # if image9_HTLU is active this frame...
                    if image9_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTLU is stopping this frame...
                    if image9_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTLU.tStop = t  # not accounting for scr refresh
                            image9_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTLU.stopped')
                            # update status
                            image9_HTLU.status = FINISHED
                            image9_HTLU.setAutoDraw(False)
                    
                    # *image10_HTLU* updates
                    
                    # if image10_HTLU is starting this frame...
                    if image10_HTLU.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTLU.frameNStart = frameN  # exact frame index
                        image10_HTLU.tStart = t  # local t and not account for scr refresh
                        image10_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTLU.started')
                        # update status
                        image10_HTLU.status = STARTED
                        image10_HTLU.setAutoDraw(True)
                    
                    # if image10_HTLU is active this frame...
                    if image10_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTLU is stopping this frame...
                    if image10_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTLU.tStop = t  # not accounting for scr refresh
                            image10_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTLU.stopped')
                            # update status
                            image10_HTLU.status = FINISHED
                            image10_HTLU.setAutoDraw(False)
                    
                    # *image11_HTLU* updates
                    
                    # if image11_HTLU is starting this frame...
                    if image11_HTLU.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTLU.frameNStart = frameN  # exact frame index
                        image11_HTLU.tStart = t  # local t and not account for scr refresh
                        image11_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTLU.started')
                        # update status
                        image11_HTLU.status = STARTED
                        image11_HTLU.setAutoDraw(True)
                    
                    # if image11_HTLU is active this frame...
                    if image11_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTLU is stopping this frame...
                    if image11_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTLU.tStop = t  # not accounting for scr refresh
                            image11_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTLU.stopped')
                            # update status
                            image11_HTLU.status = FINISHED
                            image11_HTLU.setAutoDraw(False)
                    
                    # *image12_HTLU* updates
                    
                    # if image12_HTLU is starting this frame...
                    if image12_HTLU.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTLU.frameNStart = frameN  # exact frame index
                        image12_HTLU.tStart = t  # local t and not account for scr refresh
                        image12_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTLU.started')
                        # update status
                        image12_HTLU.status = STARTED
                        image12_HTLU.setAutoDraw(True)
                    
                    # if image12_HTLU is active this frame...
                    if image12_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTLU is stopping this frame...
                    if image12_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTLU.tStop = t  # not accounting for scr refresh
                            image12_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTLU.stopped')
                            # update status
                            image12_HTLU.status = FINISHED
                            image12_HTLU.setAutoDraw(False)
                    
                    # *image13_HTLU* updates
                    
                    # if image13_HTLU is starting this frame...
                    if image13_HTLU.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTLU.frameNStart = frameN  # exact frame index
                        image13_HTLU.tStart = t  # local t and not account for scr refresh
                        image13_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTLU.started')
                        # update status
                        image13_HTLU.status = STARTED
                        image13_HTLU.setAutoDraw(True)
                    
                    # if image13_HTLU is active this frame...
                    if image13_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTLU is stopping this frame...
                    if image13_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTLU.tStop = t  # not accounting for scr refresh
                            image13_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTLU.stopped')
                            # update status
                            image13_HTLU.status = FINISHED
                            image13_HTLU.setAutoDraw(False)
                    
                    # *image14_HTLU* updates
                    
                    # if image14_HTLU is starting this frame...
                    if image14_HTLU.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTLU.frameNStart = frameN  # exact frame index
                        image14_HTLU.tStart = t  # local t and not account for scr refresh
                        image14_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTLU.started')
                        # update status
                        image14_HTLU.status = STARTED
                        image14_HTLU.setAutoDraw(True)
                    
                    # if image14_HTLU is active this frame...
                    if image14_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTLU is stopping this frame...
                    if image14_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTLU.tStop = t  # not accounting for scr refresh
                            image14_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTLU.stopped')
                            # update status
                            image14_HTLU.status = FINISHED
                            image14_HTLU.setAutoDraw(False)
                    
                    # *image15_HTLU* updates
                    
                    # if image15_HTLU is starting this frame...
                    if image15_HTLU.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTLU.frameNStart = frameN  # exact frame index
                        image15_HTLU.tStart = t  # local t and not account for scr refresh
                        image15_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTLU.started')
                        # update status
                        image15_HTLU.status = STARTED
                        image15_HTLU.setAutoDraw(True)
                    
                    # if image15_HTLU is active this frame...
                    if image15_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTLU is stopping this frame...
                    if image15_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTLU.tStop = t  # not accounting for scr refresh
                            image15_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTLU.stopped')
                            # update status
                            image15_HTLU.status = FINISHED
                            image15_HTLU.setAutoDraw(False)
                    
                    # *image16_HTLU* updates
                    
                    # if image16_HTLU is starting this frame...
                    if image16_HTLU.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTLU.frameNStart = frameN  # exact frame index
                        image16_HTLU.tStart = t  # local t and not account for scr refresh
                        image16_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTLU.started')
                        # update status
                        image16_HTLU.status = STARTED
                        image16_HTLU.setAutoDraw(True)
                    
                    # if image16_HTLU is active this frame...
                    if image16_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTLU is stopping this frame...
                    if image16_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTLU.tStop = t  # not accounting for scr refresh
                            image16_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTLU.stopped')
                            # update status
                            image16_HTLU.status = FINISHED
                            image16_HTLU.setAutoDraw(False)
                    
                    # *image17_HTLU* updates
                    
                    # if image17_HTLU is starting this frame...
                    if image17_HTLU.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTLU.frameNStart = frameN  # exact frame index
                        image17_HTLU.tStart = t  # local t and not account for scr refresh
                        image17_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTLU.started')
                        # update status
                        image17_HTLU.status = STARTED
                        image17_HTLU.setAutoDraw(True)
                    
                    # if image17_HTLU is active this frame...
                    if image17_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTLU is stopping this frame...
                    if image17_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTLU.tStop = t  # not accounting for scr refresh
                            image17_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTLU.stopped')
                            # update status
                            image17_HTLU.status = FINISHED
                            image17_HTLU.setAutoDraw(False)
                    
                    # *image18_HTLU* updates
                    
                    # if image18_HTLU is starting this frame...
                    if image18_HTLU.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTLU.frameNStart = frameN  # exact frame index
                        image18_HTLU.tStart = t  # local t and not account for scr refresh
                        image18_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTLU.started')
                        # update status
                        image18_HTLU.status = STARTED
                        image18_HTLU.setAutoDraw(True)
                    
                    # if image18_HTLU is active this frame...
                    if image18_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTLU is stopping this frame...
                    if image18_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTLU.tStop = t  # not accounting for scr refresh
                            image18_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTLU.stopped')
                            # update status
                            image18_HTLU.status = FINISHED
                            image18_HTLU.setAutoDraw(False)
                    
                    # *image19_HTLU* updates
                    
                    # if image19_HTLU is starting this frame...
                    if image19_HTLU.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTLU.frameNStart = frameN  # exact frame index
                        image19_HTLU.tStart = t  # local t and not account for scr refresh
                        image19_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTLU.started')
                        # update status
                        image19_HTLU.status = STARTED
                        image19_HTLU.setAutoDraw(True)
                    
                    # if image19_HTLU is active this frame...
                    if image19_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTLU is stopping this frame...
                    if image19_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTLU.tStop = t  # not accounting for scr refresh
                            image19_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTLU.stopped')
                            # update status
                            image19_HTLU.status = FINISHED
                            image19_HTLU.setAutoDraw(False)
                    
                    # *image20_HTLU* updates
                    
                    # if image20_HTLU is starting this frame...
                    if image20_HTLU.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTLU.frameNStart = frameN  # exact frame index
                        image20_HTLU.tStart = t  # local t and not account for scr refresh
                        image20_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTLU.started')
                        # update status
                        image20_HTLU.status = STARTED
                        image20_HTLU.setAutoDraw(True)
                    
                    # if image20_HTLU is active this frame...
                    if image20_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTLU is stopping this frame...
                    if image20_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTLU.tStop = t  # not accounting for scr refresh
                            image20_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTLU.stopped')
                            # update status
                            image20_HTLU.status = FINISHED
                            image20_HTLU.setAutoDraw(False)
                    
                    # *image21_HTLU* updates
                    
                    # if image21_HTLU is starting this frame...
                    if image21_HTLU.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTLU.frameNStart = frameN  # exact frame index
                        image21_HTLU.tStart = t  # local t and not account for scr refresh
                        image21_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTLU.started')
                        # update status
                        image21_HTLU.status = STARTED
                        image21_HTLU.setAutoDraw(True)
                    
                    # if image21_HTLU is active this frame...
                    if image21_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTLU is stopping this frame...
                    if image21_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTLU.tStop = t  # not accounting for scr refresh
                            image21_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTLU.stopped')
                            # update status
                            image21_HTLU.status = FINISHED
                            image21_HTLU.setAutoDraw(False)
                    
                    # *image22_HTLU* updates
                    
                    # if image22_HTLU is starting this frame...
                    if image22_HTLU.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTLU.frameNStart = frameN  # exact frame index
                        image22_HTLU.tStart = t  # local t and not account for scr refresh
                        image22_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTLU.started')
                        # update status
                        image22_HTLU.status = STARTED
                        image22_HTLU.setAutoDraw(True)
                    
                    # if image22_HTLU is active this frame...
                    if image22_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTLU is stopping this frame...
                    if image22_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTLU.tStop = t  # not accounting for scr refresh
                            image22_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTLU.stopped')
                            # update status
                            image22_HTLU.status = FINISHED
                            image22_HTLU.setAutoDraw(False)
                    
                    # *image23_HTLU* updates
                    
                    # if image23_HTLU is starting this frame...
                    if image23_HTLU.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTLU.frameNStart = frameN  # exact frame index
                        image23_HTLU.tStart = t  # local t and not account for scr refresh
                        image23_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTLU.started')
                        # update status
                        image23_HTLU.status = STARTED
                        image23_HTLU.setAutoDraw(True)
                    
                    # if image23_HTLU is active this frame...
                    if image23_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTLU is stopping this frame...
                    if image23_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTLU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTLU.tStop = t  # not accounting for scr refresh
                            image23_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTLU.stopped')
                            # update status
                            image23_HTLU.status = FINISHED
                            image23_HTLU.setAutoDraw(False)
                    
                    # *image24_HTLU* updates
                    
                    # if image24_HTLU is starting this frame...
                    if image24_HTLU.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTLU.frameNStart = frameN  # exact frame index
                        image24_HTLU.tStart = t  # local t and not account for scr refresh
                        image24_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTLU.started')
                        # update status
                        image24_HTLU.status = STARTED
                        image24_HTLU.setAutoDraw(True)
                    
                    # if image24_HTLU is active this frame...
                    if image24_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTLU is stopping this frame...
                    if image24_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTLU.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTLU.tStop = t  # not accounting for scr refresh
                            image24_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTLU.stopped')
                            # update status
                            image24_HTLU.status = FINISHED
                            image24_HTLU.setAutoDraw(False)
                    
                    # *image_Frame_HTLU* updates
                    
                    # if image_Frame_HTLU is starting this frame...
                    if image_Frame_HTLU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTLU.frameNStart = frameN  # exact frame index
                        image_Frame_HTLU.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTLU.started')
                        # update status
                        image_Frame_HTLU.status = STARTED
                        image_Frame_HTLU.setAutoDraw(True)
                    
                    # if image_Frame_HTLU is active this frame...
                    if image_Frame_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTLU is stopping this frame...
                    if image_Frame_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTLU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTLU.tStop = t  # not accounting for scr refresh
                            image_Frame_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTLU.stopped')
                            # update status
                            image_Frame_HTLU.status = FINISHED
                            image_Frame_HTLU.setAutoDraw(False)
                    
                    # *image_Cue_HTLU* updates
                    
                    # if image_Cue_HTLU is starting this frame...
                    if image_Cue_HTLU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTLU.frameNStart = frameN  # exact frame index
                        image_Cue_HTLU.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTLU.started')
                        # update status
                        image_Cue_HTLU.status = STARTED
                        image_Cue_HTLU.setAutoDraw(True)
                    
                    # if image_Cue_HTLU is active this frame...
                    if image_Cue_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTLU is stopping this frame...
                    if image_Cue_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTLU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTLU.tStop = t  # not accounting for scr refresh
                            image_Cue_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTLU.stopped')
                            # update status
                            image_Cue_HTLU.status = FINISHED
                            image_Cue_HTLU.setAutoDraw(False)
                    
                    # *image_Cross_HTLU* updates
                    
                    # if image_Cross_HTLU is starting this frame...
                    if image_Cross_HTLU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTLU.frameNStart = frameN  # exact frame index
                        image_Cross_HTLU.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTLU.started')
                        # update status
                        image_Cross_HTLU.status = STARTED
                        image_Cross_HTLU.setAutoDraw(True)
                    
                    # if image_Cross_HTLU is active this frame...
                    if image_Cross_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTLU is stopping this frame...
                    if image_Cross_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTLU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTLU.tStop = t  # not accounting for scr refresh
                            image_Cross_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTLU.stopped')
                            # update status
                            image_Cross_HTLU.status = FINISHED
                            image_Cross_HTLU.setAutoDraw(False)
                    
                    # *image_Tar_HTLU* updates
                    
                    # if image_Tar_HTLU is starting this frame...
                    if image_Tar_HTLU.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTLU.frameNStart = frameN  # exact frame index
                        image_Tar_HTLU.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTLU.started')
                        # update status
                        image_Tar_HTLU.status = STARTED
                        image_Tar_HTLU.setAutoDraw(True)
                    
                    # if image_Tar_HTLU is active this frame...
                    if image_Tar_HTLU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTLU is stopping this frame...
                    if image_Tar_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTLU.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END
                            # keep track of stop time/frame for later
                            image_Tar_HTLU.tStop = t  # not accounting for scr refresh
                            image_Tar_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTLU.stopped')
                            # update status
                            image_Tar_HTLU.status = FINISHED
                            image_Tar_HTLU.setAutoDraw(False)
                    
                    # *key_HTLU* updates
                    waitOnFlip = False
                    
                    # if key_HTLU is starting this frame...
                    if key_HTLU.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTLU.frameNStart = frameN  # exact frame index
                        key_HTLU.tStart = t  # local t and not account for scr refresh
                        key_HTLU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTLU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTLU.started')
                        # update status
                        key_HTLU.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTLU.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTLU.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTLU is stopping this frame...
                    if key_HTLU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTLU.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTLU.tStop = t  # not accounting for scr refresh
                            key_HTLU.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTLU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTLU.stopped')
                            # update status
                            key_HTLU.status = FINISHED
                            key_HTLU.status = FINISHED
                    if key_HTLU.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTLU.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTLU_allKeys.extend(theseKeys)
                        if len(_key_HTLU_allKeys):
                            key_HTLU.keys = _key_HTLU_allKeys[0].name  # just the first key pressed
                            key_HTLU.rt = _key_HTLU_allKeys[0].rt
                            key_HTLU.duration = _key_HTLU_allKeys[0].duration
                            # was this correct?
                            ### CUSTOM START ###
                            #if (key_HTLU.keys == str(corr_resp)) or (key_HTLU.keys == corr_resp):
                            if (key_HTLU.keys == str(my_corrResp_R[my_outerloopcounter][my_TDAR_counter])) or (key_HTLU.keys == my_corrResp_R[my_outerloopcounter][my_TDAR_counter]): ## CUSTOM: Replaced corr_resp with own variable my_corrResp_*
                                key_HTLU.corr = 1
                            else:
                                key_HTLU.corr = 0
                            ### CUSTOM END ###
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTLU.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTLU.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END
                
                # --- Ending Routine "HTLU" ---
                for thisComponent in HTLU.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTLU
                HTLU.tStop = globalClock.getTime(format='float')
                HTLU.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTLU.stopped', HTLU.tStop)
                # check responses
                if key_HTLU.keys in ['', [], None]:  # No response was made
                    key_HTLU.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_R[my_outerloopcounter][my_TDAR_counter]).lower() == 'none': ## CUSTOM START & END
                       key_HTLU.corr = 1;  # correct non-response
                    else:
                       key_HTLU.corr = 0;  # failed to respond (incorrectly)
                # store data for Trials_HTLU (TrialHandler)
                Trials_HTLU.addData('key_HTLU.keys',key_HTLU.keys)
                Trials_HTLU.addData('key_HTLU.corr', key_HTLU.corr)
                if key_HTLU.keys != None:  # we had a response
                    Trials_HTLU.addData('key_HTLU.rt', key_HTLU.rt)
                    Trials_HTLU.addData('key_HTLU.duration', key_HTLU.duration)
                try:
                    if image1_HTLU.tStopRefresh is not None:
                        duration_val = image1_HTLU.tStopRefresh - image1_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image1_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTLU).__name__,
                        trial_type='image1_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image1_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTLU.tStopRefresh is not None:
                        duration_val = image2_HTLU.tStopRefresh - image2_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image2_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTLU).__name__,
                        trial_type='image2_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image2_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTLU.tStopRefresh is not None:
                        duration_val = image3_HTLU.tStopRefresh - image3_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image3_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTLU).__name__,
                        trial_type='image3_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image3_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTLU.tStopRefresh is not None:
                        duration_val = image4_HTLU.tStopRefresh - image4_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image4_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTLU).__name__,
                        trial_type='image4_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image4_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTLU.tStopRefresh is not None:
                        duration_val = image5_HTLU.tStopRefresh - image5_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image5_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTLU).__name__,
                        trial_type='image5_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image5_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTLU.tStopRefresh is not None:
                        duration_val = image6_HTLU.tStopRefresh - image6_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image6_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTLU).__name__,
                        trial_type='image6_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image6_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTLU.tStopRefresh is not None:
                        duration_val = image7_HTLU.tStopRefresh - image7_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image7_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTLU).__name__,
                        trial_type='image7_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image7_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTLU.tStopRefresh is not None:
                        duration_val = image8_HTLU.tStopRefresh - image8_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image8_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTLU).__name__,
                        trial_type='image8_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image8_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTLU.tStopRefresh is not None:
                        duration_val = image9_HTLU.tStopRefresh - image9_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image9_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTLU).__name__,
                        trial_type='image9_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image9_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTLU.tStopRefresh is not None:
                        duration_val = image10_HTLU.tStopRefresh - image10_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image10_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTLU).__name__,
                        trial_type='image10_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image10_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTLU.tStopRefresh is not None:
                        duration_val = image11_HTLU.tStopRefresh - image11_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image11_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTLU).__name__,
                        trial_type='image11_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image11_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTLU.tStopRefresh is not None:
                        duration_val = image12_HTLU.tStopRefresh - image12_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image12_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTLU).__name__,
                        trial_type='image12_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image12_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTLU.tStopRefresh is not None:
                        duration_val = image13_HTLU.tStopRefresh - image13_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image13_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTLU).__name__,
                        trial_type='image13_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image13_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTLU.tStopRefresh is not None:
                        duration_val = image14_HTLU.tStopRefresh - image14_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image14_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTLU).__name__,
                        trial_type='image14_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image14_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTLU.tStopRefresh is not None:
                        duration_val = image15_HTLU.tStopRefresh - image15_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image15_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTLU).__name__,
                        trial_type='image15_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image15_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTLU.tStopRefresh is not None:
                        duration_val = image16_HTLU.tStopRefresh - image16_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image16_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTLU).__name__,
                        trial_type='image16_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image16_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTLU.tStopRefresh is not None:
                        duration_val = image17_HTLU.tStopRefresh - image17_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image17_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTLU).__name__,
                        trial_type='image17_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image17_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTLU.tStopRefresh is not None:
                        duration_val = image18_HTLU.tStopRefresh - image18_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image18_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTLU).__name__,
                        trial_type='image18_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image18_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTLU.tStopRefresh is not None:
                        duration_val = image19_HTLU.tStopRefresh - image19_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image19_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTLU).__name__,
                        trial_type='image19_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image19_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTLU.tStopRefresh is not None:
                        duration_val = image20_HTLU.tStopRefresh - image20_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image20_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTLU).__name__,
                        trial_type='image20_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image20_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTLU.tStopRefresh is not None:
                        duration_val = image21_HTLU.tStopRefresh - image21_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image21_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTLU).__name__,
                        trial_type='image21_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image21_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTLU.tStopRefresh is not None:
                        duration_val = image22_HTLU.tStopRefresh - image22_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image22_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTLU).__name__,
                        trial_type='image22_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image22_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTLU.tStopRefresh is not None:
                        duration_val = image23_HTLU.tStopRefresh - image23_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image23_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTLU).__name__,
                        trial_type='image23_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image23_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTLU.tStopRefresh is not None:
                        duration_val = image24_HTLU.tStopRefresh - image24_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image24_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTLU).__name__,
                        trial_type='image24_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image24_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTLU.tStopRefresh is not None:
                        duration_val = image_Cue_HTLU.tStopRefresh - image_Cue_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image_Cue_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTLU).__name__,
                        trial_type='image_Cue_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image_Cue_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTLU.tStopRefresh is not None:
                        duration_val = image_Tar_HTLU.tStopRefresh - image_Tar_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - image_Tar_HTLU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTLU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTLU).__name__,
                        trial_type='image_Tar_HTLU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_image_Tar_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTLU.tStopRefresh is not None:
                        duration_val = key_HTLU.tStopRefresh - key_HTLU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLU.stopped'] - key_HTLU.tStartRefresh
                    if hasattr(key_HTLU, 'rt'):
                        rt_val = key_HTLU.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTLU" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTLU.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTLU).__name__,
                        trial_type='key_HTLU',
                        value=key_HTLU.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTLU.addData('bidsEvent_key_HTLU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTLU.maxDurationReached:
                    routineTimer.addTime(-HTLU.maxDuration)
                elif HTLU.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                #--- CUSTOM START ---
                my_TDAR_counter = my_TDAR_counter + 1
                #--- CUSTOM END ---
                
            # completed 1.0 repeats of 'Trials_HTLU'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if Trials_HTLU.trialList in ([], [None], None):
                params = []
            else:
                params = Trials_HTLU.trialList[0].keys()
            # save data for this loop
            Trials_HTLU.saveAsExcel(filename + '.xlsx', sheetName='Trials_HTLU',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            Trials_HTLU.saveAsText(filename + 'Trials_HTLU.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
        #Custom-Comment: End TDA_R
        elif my_seq_num[my_outerloopcounter]==3:
        #Custom-Comment: Start TDA_GL
        
            # set up handler to look after randomisation of conditions etc
            Trials_HTRF = data.TrialHandler2(
                name='Trials_HTRF',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TR.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(Trials_HTRF)  # add the loop to the experiment
            thisTrials_HTRF = Trials_HTRF.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRF.rgb)
            if thisTrials_HTRF != None:
                for paramName in thisTrials_HTRF:
                    globals()[paramName] = thisTrials_HTRF[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDAGL_counter = 0
            #--- CUSTOM END ---
            
            for thisTrials_HTRF in Trials_HTRF:
                currentLoop = Trials_HTRF
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRF.rgb)
                if thisTrials_HTRF != None:
                    for paramName in thisTrials_HTRF:
                        globals()[paramName] = thisTrials_HTRF[paramName]
                
                # --- Prepare to start Routine "HTRF" ---
                # create an object to store info about Routine HTRF
                HTRF = data.Routine(
                    name='HTRF',
                    components=[image1_HTRF, image2_HTRF, image3_HTRF, image4_HTRF, image5_HTRF, image6_HTRF, image7_HTRF, image8_HTRF, image9_HTRF, image10_HTRF, image11_HTRF, image12_HTRF, image13_HTRF, image14_HTRF, image15_HTRF, image16_HTRF, image17_HTRF, image18_HTRF, image19_HTRF, image20_HTRF, image21_HTRF, image22_HTRF, image23_HTRF, image24_HTRF, image_Frame_HTRF, image_Cue_HTRF, image_Cross_HTRF, image_Tar_HTRF, key_HTRF],
                )
                HTRF.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                #--- CUSTOM START ---
                image_Tar_HTRF.setPos((my_coordTDA_GL[0, my_TDAGL_counter, my_outerloopcounter], my_coordTDA_GL[1, my_TDAGL_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTRF.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTRF
                key_HTRF.keys = []
                key_HTRF.rt = []
                _key_HTRF_allKeys = []
                # store start times for HTRF
                HTRF.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTRF.tStart = globalClock.getTime(format='float')
                HTRF.status = STARTED
                thisExp.addData('HTRF.started', HTRF.tStart)
                HTRF.maxDuration = None
                # keep track of which components have finished
                HTRFComponents = HTRF.components
                for thisComponent in HTRF.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTRF" ---
                # if trial has changed, end Routine now
                if isinstance(Trials_HTRF, data.TrialHandler2) and thisTrials_HTRF.thisN != Trials_HTRF.thisTrial.thisN:
                    continueRoutine = False
                HTRF.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTRF* updates
                    
                    # if image1_HTRF is starting this frame...
                    if image1_HTRF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTRF.frameNStart = frameN  # exact frame index
                        image1_HTRF.tStart = t  # local t and not account for scr refresh
                        image1_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTRF.started')
                        # update status
                        image1_HTRF.status = STARTED
                        image1_HTRF.setAutoDraw(True)
                    
                    # if image1_HTRF is active this frame...
                    if image1_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTRF is stopping this frame...
                    if image1_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTRF.tStop = t  # not accounting for scr refresh
                            image1_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTRF.stopped')
                            # update status
                            image1_HTRF.status = FINISHED
                            image1_HTRF.setAutoDraw(False)
                    
                    # *image2_HTRF* updates
                    
                    # if image2_HTRF is starting this frame...
                    if image2_HTRF.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTRF.frameNStart = frameN  # exact frame index
                        image2_HTRF.tStart = t  # local t and not account for scr refresh
                        image2_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTRF.started')
                        # update status
                        image2_HTRF.status = STARTED
                        image2_HTRF.setAutoDraw(True)
                    
                    # if image2_HTRF is active this frame...
                    if image2_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTRF is stopping this frame...
                    if image2_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTRF.tStop = t  # not accounting for scr refresh
                            image2_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTRF.stopped')
                            # update status
                            image2_HTRF.status = FINISHED
                            image2_HTRF.setAutoDraw(False)
                    
                    # *image3_HTRF* updates
                    
                    # if image3_HTRF is starting this frame...
                    if image3_HTRF.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTRF.frameNStart = frameN  # exact frame index
                        image3_HTRF.tStart = t  # local t and not account for scr refresh
                        image3_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTRF.started')
                        # update status
                        image3_HTRF.status = STARTED
                        image3_HTRF.setAutoDraw(True)
                    
                    # if image3_HTRF is active this frame...
                    if image3_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTRF is stopping this frame...
                    if image3_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTRF.tStop = t  # not accounting for scr refresh
                            image3_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTRF.stopped')
                            # update status
                            image3_HTRF.status = FINISHED
                            image3_HTRF.setAutoDraw(False)
                    
                    # *image4_HTRF* updates
                    
                    # if image4_HTRF is starting this frame...
                    if image4_HTRF.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTRF.frameNStart = frameN  # exact frame index
                        image4_HTRF.tStart = t  # local t and not account for scr refresh
                        image4_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTRF.started')
                        # update status
                        image4_HTRF.status = STARTED
                        image4_HTRF.setAutoDraw(True)
                    
                    # if image4_HTRF is active this frame...
                    if image4_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTRF is stopping this frame...
                    if image4_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTRF.tStop = t  # not accounting for scr refresh
                            image4_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTRF.stopped')
                            # update status
                            image4_HTRF.status = FINISHED
                            image4_HTRF.setAutoDraw(False)
                    
                    # *image5_HTRF* updates
                    
                    # if image5_HTRF is starting this frame...
                    if image5_HTRF.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTRF.frameNStart = frameN  # exact frame index
                        image5_HTRF.tStart = t  # local t and not account for scr refresh
                        image5_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTRF.started')
                        # update status
                        image5_HTRF.status = STARTED
                        image5_HTRF.setAutoDraw(True)
                    
                    # if image5_HTRF is active this frame...
                    if image5_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTRF is stopping this frame...
                    if image5_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTRF.tStop = t  # not accounting for scr refresh
                            image5_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTRF.stopped')
                            # update status
                            image5_HTRF.status = FINISHED
                            image5_HTRF.setAutoDraw(False)
                    
                    # *image6_HTRF* updates
                    
                    # if image6_HTRF is starting this frame...
                    if image6_HTRF.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTRF.frameNStart = frameN  # exact frame index
                        image6_HTRF.tStart = t  # local t and not account for scr refresh
                        image6_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTRF.started')
                        # update status
                        image6_HTRF.status = STARTED
                        image6_HTRF.setAutoDraw(True)
                    
                    # if image6_HTRF is active this frame...
                    if image6_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTRF is stopping this frame...
                    if image6_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTRF.tStop = t  # not accounting for scr refresh
                            image6_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTRF.stopped')
                            # update status
                            image6_HTRF.status = FINISHED
                            image6_HTRF.setAutoDraw(False)
                    
                    # *image7_HTRF* updates
                    
                    # if image7_HTRF is starting this frame...
                    if image7_HTRF.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTRF.frameNStart = frameN  # exact frame index
                        image7_HTRF.tStart = t  # local t and not account for scr refresh
                        image7_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTRF.started')
                        # update status
                        image7_HTRF.status = STARTED
                        image7_HTRF.setAutoDraw(True)
                    
                    # if image7_HTRF is active this frame...
                    if image7_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTRF is stopping this frame...
                    if image7_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTRF.tStop = t  # not accounting for scr refresh
                            image7_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTRF.stopped')
                            # update status
                            image7_HTRF.status = FINISHED
                            image7_HTRF.setAutoDraw(False)
                    
                    # *image8_HTRF* updates
                    
                    # if image8_HTRF is starting this frame...
                    if image8_HTRF.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTRF.frameNStart = frameN  # exact frame index
                        image8_HTRF.tStart = t  # local t and not account for scr refresh
                        image8_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTRF.started')
                        # update status
                        image8_HTRF.status = STARTED
                        image8_HTRF.setAutoDraw(True)
                    
                    # if image8_HTRF is active this frame...
                    if image8_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTRF is stopping this frame...
                    if image8_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTRF.tStop = t  # not accounting for scr refresh
                            image8_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTRF.stopped')
                            # update status
                            image8_HTRF.status = FINISHED
                            image8_HTRF.setAutoDraw(False)
                    
                    # *image9_HTRF* updates
                    
                    # if image9_HTRF is starting this frame...
                    if image9_HTRF.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTRF.frameNStart = frameN  # exact frame index
                        image9_HTRF.tStart = t  # local t and not account for scr refresh
                        image9_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTRF.started')
                        # update status
                        image9_HTRF.status = STARTED
                        image9_HTRF.setAutoDraw(True)
                    
                    # if image9_HTRF is active this frame...
                    if image9_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTRF is stopping this frame...
                    if image9_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTRF.tStop = t  # not accounting for scr refresh
                            image9_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTRF.stopped')
                            # update status
                            image9_HTRF.status = FINISHED
                            image9_HTRF.setAutoDraw(False)
                    
                    # *image10_HTRF* updates
                    
                    # if image10_HTRF is starting this frame...
                    if image10_HTRF.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTRF.frameNStart = frameN  # exact frame index
                        image10_HTRF.tStart = t  # local t and not account for scr refresh
                        image10_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTRF.started')
                        # update status
                        image10_HTRF.status = STARTED
                        image10_HTRF.setAutoDraw(True)
                    
                    # if image10_HTRF is active this frame...
                    if image10_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTRF is stopping this frame...
                    if image10_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTRF.tStop = t  # not accounting for scr refresh
                            image10_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTRF.stopped')
                            # update status
                            image10_HTRF.status = FINISHED
                            image10_HTRF.setAutoDraw(False)
                    
                    # *image11_HTRF* updates
                    
                    # if image11_HTRF is starting this frame...
                    if image11_HTRF.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTRF.frameNStart = frameN  # exact frame index
                        image11_HTRF.tStart = t  # local t and not account for scr refresh
                        image11_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTRF.started')
                        # update status
                        image11_HTRF.status = STARTED
                        image11_HTRF.setAutoDraw(True)
                    
                    # if image11_HTRF is active this frame...
                    if image11_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTRF is stopping this frame...
                    if image11_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTRF.tStop = t  # not accounting for scr refresh
                            image11_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTRF.stopped')
                            # update status
                            image11_HTRF.status = FINISHED
                            image11_HTRF.setAutoDraw(False)
                    
                    # *image12_HTRF* updates
                    
                    # if image12_HTRF is starting this frame...
                    if image12_HTRF.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTRF.frameNStart = frameN  # exact frame index
                        image12_HTRF.tStart = t  # local t and not account for scr refresh
                        image12_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTRF.started')
                        # update status
                        image12_HTRF.status = STARTED
                        image12_HTRF.setAutoDraw(True)
                    
                    # if image12_HTRF is active this frame...
                    if image12_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTRF is stopping this frame...
                    if image12_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTRF.tStop = t  # not accounting for scr refresh
                            image12_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTRF.stopped')
                            # update status
                            image12_HTRF.status = FINISHED
                            image12_HTRF.setAutoDraw(False)
                    
                    # *image13_HTRF* updates
                    
                    # if image13_HTRF is starting this frame...
                    if image13_HTRF.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTRF.frameNStart = frameN  # exact frame index
                        image13_HTRF.tStart = t  # local t and not account for scr refresh
                        image13_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTRF.started')
                        # update status
                        image13_HTRF.status = STARTED
                        image13_HTRF.setAutoDraw(True)
                    
                    # if image13_HTRF is active this frame...
                    if image13_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTRF is stopping this frame...
                    if image13_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTRF.tStop = t  # not accounting for scr refresh
                            image13_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTRF.stopped')
                            # update status
                            image13_HTRF.status = FINISHED
                            image13_HTRF.setAutoDraw(False)
                    
                    # *image14_HTRF* updates
                    
                    # if image14_HTRF is starting this frame...
                    if image14_HTRF.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTRF.frameNStart = frameN  # exact frame index
                        image14_HTRF.tStart = t  # local t and not account for scr refresh
                        image14_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTRF.started')
                        # update status
                        image14_HTRF.status = STARTED
                        image14_HTRF.setAutoDraw(True)
                    
                    # if image14_HTRF is active this frame...
                    if image14_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTRF is stopping this frame...
                    if image14_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTRF.tStop = t  # not accounting for scr refresh
                            image14_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTRF.stopped')
                            # update status
                            image14_HTRF.status = FINISHED
                            image14_HTRF.setAutoDraw(False)
                    
                    # *image15_HTRF* updates
                    
                    # if image15_HTRF is starting this frame...
                    if image15_HTRF.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTRF.frameNStart = frameN  # exact frame index
                        image15_HTRF.tStart = t  # local t and not account for scr refresh
                        image15_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTRF.started')
                        # update status
                        image15_HTRF.status = STARTED
                        image15_HTRF.setAutoDraw(True)
                    
                    # if image15_HTRF is active this frame...
                    if image15_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTRF is stopping this frame...
                    if image15_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTRF.tStop = t  # not accounting for scr refresh
                            image15_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTRF.stopped')
                            # update status
                            image15_HTRF.status = FINISHED
                            image15_HTRF.setAutoDraw(False)
                    
                    # *image16_HTRF* updates
                    
                    # if image16_HTRF is starting this frame...
                    if image16_HTRF.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTRF.frameNStart = frameN  # exact frame index
                        image16_HTRF.tStart = t  # local t and not account for scr refresh
                        image16_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTRF.started')
                        # update status
                        image16_HTRF.status = STARTED
                        image16_HTRF.setAutoDraw(True)
                    
                    # if image16_HTRF is active this frame...
                    if image16_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTRF is stopping this frame...
                    if image16_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTRF.tStop = t  # not accounting for scr refresh
                            image16_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTRF.stopped')
                            # update status
                            image16_HTRF.status = FINISHED
                            image16_HTRF.setAutoDraw(False)
                    
                    # *image17_HTRF* updates
                    
                    # if image17_HTRF is starting this frame...
                    if image17_HTRF.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTRF.frameNStart = frameN  # exact frame index
                        image17_HTRF.tStart = t  # local t and not account for scr refresh
                        image17_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTRF.started')
                        # update status
                        image17_HTRF.status = STARTED
                        image17_HTRF.setAutoDraw(True)
                    
                    # if image17_HTRF is active this frame...
                    if image17_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTRF is stopping this frame...
                    if image17_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTRF.tStop = t  # not accounting for scr refresh
                            image17_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTRF.stopped')
                            # update status
                            image17_HTRF.status = FINISHED
                            image17_HTRF.setAutoDraw(False)
                    
                    # *image18_HTRF* updates
                    
                    # if image18_HTRF is starting this frame...
                    if image18_HTRF.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTRF.frameNStart = frameN  # exact frame index
                        image18_HTRF.tStart = t  # local t and not account for scr refresh
                        image18_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTRF.started')
                        # update status
                        image18_HTRF.status = STARTED
                        image18_HTRF.setAutoDraw(True)
                    
                    # if image18_HTRF is active this frame...
                    if image18_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTRF is stopping this frame...
                    if image18_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTRF.tStop = t  # not accounting for scr refresh
                            image18_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTRF.stopped')
                            # update status
                            image18_HTRF.status = FINISHED
                            image18_HTRF.setAutoDraw(False)
                    
                    # *image19_HTRF* updates
                    
                    # if image19_HTRF is starting this frame...
                    if image19_HTRF.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTRF.frameNStart = frameN  # exact frame index
                        image19_HTRF.tStart = t  # local t and not account for scr refresh
                        image19_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTRF.started')
                        # update status
                        image19_HTRF.status = STARTED
                        image19_HTRF.setAutoDraw(True)
                    
                    # if image19_HTRF is active this frame...
                    if image19_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTRF is stopping this frame...
                    if image19_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTRF.tStop = t  # not accounting for scr refresh
                            image19_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTRF.stopped')
                            # update status
                            image19_HTRF.status = FINISHED
                            image19_HTRF.setAutoDraw(False)
                    
                    # *image20_HTRF* updates
                    
                    # if image20_HTRF is starting this frame...
                    if image20_HTRF.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTRF.frameNStart = frameN  # exact frame index
                        image20_HTRF.tStart = t  # local t and not account for scr refresh
                        image20_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTRF.started')
                        # update status
                        image20_HTRF.status = STARTED
                        image20_HTRF.setAutoDraw(True)
                    
                    # if image20_HTRF is active this frame...
                    if image20_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTRF is stopping this frame...
                    if image20_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTRF.tStop = t  # not accounting for scr refresh
                            image20_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTRF.stopped')
                            # update status
                            image20_HTRF.status = FINISHED
                            image20_HTRF.setAutoDraw(False)
                    
                    # *image21_HTRF* updates
                    
                    # if image21_HTRF is starting this frame...
                    if image21_HTRF.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTRF.frameNStart = frameN  # exact frame index
                        image21_HTRF.tStart = t  # local t and not account for scr refresh
                        image21_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTRF.started')
                        # update status
                        image21_HTRF.status = STARTED
                        image21_HTRF.setAutoDraw(True)
                    
                    # if image21_HTRF is active this frame...
                    if image21_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTRF is stopping this frame...
                    if image21_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTRF.tStop = t  # not accounting for scr refresh
                            image21_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTRF.stopped')
                            # update status
                            image21_HTRF.status = FINISHED
                            image21_HTRF.setAutoDraw(False)
                    
                    # *image22_HTRF* updates
                    
                    # if image22_HTRF is starting this frame...
                    if image22_HTRF.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTRF.frameNStart = frameN  # exact frame index
                        image22_HTRF.tStart = t  # local t and not account for scr refresh
                        image22_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTRF.started')
                        # update status
                        image22_HTRF.status = STARTED
                        image22_HTRF.setAutoDraw(True)
                    
                    # if image22_HTRF is active this frame...
                    if image22_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTRF is stopping this frame...
                    if image22_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTRF.tStop = t  # not accounting for scr refresh
                            image22_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTRF.stopped')
                            # update status
                            image22_HTRF.status = FINISHED
                            image22_HTRF.setAutoDraw(False)
                    
                    # *image23_HTRF* updates
                    
                    # if image23_HTRF is starting this frame...
                    if image23_HTRF.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTRF.frameNStart = frameN  # exact frame index
                        image23_HTRF.tStart = t  # local t and not account for scr refresh
                        image23_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTRF.started')
                        # update status
                        image23_HTRF.status = STARTED
                        image23_HTRF.setAutoDraw(True)
                    
                    # if image23_HTRF is active this frame...
                    if image23_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTRF is stopping this frame...
                    if image23_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTRF.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTRF.tStop = t  # not accounting for scr refresh
                            image23_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTRF.stopped')
                            # update status
                            image23_HTRF.status = FINISHED
                            image23_HTRF.setAutoDraw(False)
                    
                    # *image24_HTRF* updates
                    
                    # if image24_HTRF is starting this frame...
                    if image24_HTRF.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTRF.frameNStart = frameN  # exact frame index
                        image24_HTRF.tStart = t  # local t and not account for scr refresh
                        image24_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTRF.started')
                        # update status
                        image24_HTRF.status = STARTED
                        image24_HTRF.setAutoDraw(True)
                    
                    # if image24_HTRF is active this frame...
                    if image24_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTRF is stopping this frame...
                    if image24_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTRF.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTRF.tStop = t  # not accounting for scr refresh
                            image24_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTRF.stopped')
                            # update status
                            image24_HTRF.status = FINISHED
                            image24_HTRF.setAutoDraw(False)
                    
                    # *image_Frame_HTRF* updates
                    
                    # if image_Frame_HTRF is starting this frame...
                    if image_Frame_HTRF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTRF.frameNStart = frameN  # exact frame index
                        image_Frame_HTRF.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTRF.started')
                        # update status
                        image_Frame_HTRF.status = STARTED
                        image_Frame_HTRF.setAutoDraw(True)
                    
                    # if image_Frame_HTRF is active this frame...
                    if image_Frame_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTRF is stopping this frame...
                    if image_Frame_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTRF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTRF.tStop = t  # not accounting for scr refresh
                            image_Frame_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTRF.stopped')
                            # update status
                            image_Frame_HTRF.status = FINISHED
                            image_Frame_HTRF.setAutoDraw(False)
                    
                    # *image_Cue_HTRF* updates
                    
                    # if image_Cue_HTRF is starting this frame...
                    if image_Cue_HTRF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTRF.frameNStart = frameN  # exact frame index
                        image_Cue_HTRF.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTRF.started')
                        # update status
                        image_Cue_HTRF.status = STARTED
                        image_Cue_HTRF.setAutoDraw(True)
                    
                    # if image_Cue_HTRF is active this frame...
                    if image_Cue_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTRF is stopping this frame...
                    if image_Cue_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTRF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTRF.tStop = t  # not accounting for scr refresh
                            image_Cue_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTRF.stopped')
                            # update status
                            image_Cue_HTRF.status = FINISHED
                            image_Cue_HTRF.setAutoDraw(False)
                    
                    # *image_Cross_HTRF* updates
                    
                    # if image_Cross_HTRF is starting this frame...
                    if image_Cross_HTRF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTRF.frameNStart = frameN  # exact frame index
                        image_Cross_HTRF.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTRF.started')
                        # update status
                        image_Cross_HTRF.status = STARTED
                        image_Cross_HTRF.setAutoDraw(True)
                    
                    # if image_Cross_HTRF is active this frame...
                    if image_Cross_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTRF is stopping this frame...
                    if image_Cross_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTRF.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTRF.tStop = t  # not accounting for scr refresh
                            image_Cross_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTRF.stopped')
                            # update status
                            image_Cross_HTRF.status = FINISHED
                            image_Cross_HTRF.setAutoDraw(False)
                    
                    # *image_Tar_HTRF* updates
                    
                    # if image_Tar_HTRF is starting this frame...
                    if image_Tar_HTRF.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTRF.frameNStart = frameN  # exact frame index
                        image_Tar_HTRF.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTRF.started')
                        # update status
                        image_Tar_HTRF.status = STARTED
                        image_Tar_HTRF.setAutoDraw(True)
                    
                    # if image_Tar_HTRF is active this frame...
                    if image_Tar_HTRF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTRF is stopping this frame...
                    if image_Tar_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTRF.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END 
                            # keep track of stop time/frame for later
                            image_Tar_HTRF.tStop = t  # not accounting for scr refresh
                            image_Tar_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTRF.stopped')
                            # update status
                            image_Tar_HTRF.status = FINISHED
                            image_Tar_HTRF.setAutoDraw(False)
                    
                    # *key_HTRF* updates
                    waitOnFlip = False
                    
                    # if key_HTRF is starting this frame...
                    if key_HTRF.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTRF.frameNStart = frameN  # exact frame index
                        key_HTRF.tStart = t  # local t and not account for scr refresh
                        key_HTRF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTRF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTRF.started')
                        # update status
                        key_HTRF.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTRF.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTRF.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTRF is stopping this frame...
                    if key_HTRF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTRF.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTRF.tStop = t  # not accounting for scr refresh
                            key_HTRF.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTRF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTRF.stopped')
                            # update status
                            key_HTRF.status = FINISHED
                            key_HTRF.status = FINISHED
                    if key_HTRF.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTRF.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTRF_allKeys.extend(theseKeys)
                        if len(_key_HTRF_allKeys):
                            key_HTRF.keys = _key_HTRF_allKeys[0].name  # just the first key pressed
                            key_HTRF.rt = _key_HTRF_allKeys[0].rt
                            key_HTRF.duration = _key_HTRF_allKeys[0].duration
                            # was this correct?
                            ### CUSTOM START 
                            if (key_HTRF.keys == str(my_corrResp_GL[my_outerloopcounter][my_TDAGL_counter])) or (key_HTRF.keys == my_corrResp_GL[my_outerloopcounter][my_TDAGL_counter]):
                            #if (key_HTRF.keys == str(corr_resp)) or (key_HTRF.keys == corr_resp):
                                key_HTRF.corr = 1
                            else:
                                key_HTRF.corr = 0
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTRF.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTRF.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END
                
                # --- Ending Routine "HTRF" ---
                for thisComponent in HTRF.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTRF
                HTRF.tStop = globalClock.getTime(format='float')
                HTRF.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTRF.stopped', HTRF.tStop)
                # check responses
                if key_HTRF.keys in ['', [], None]:  # No response was made
                    key_HTRF.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_GL[my_outerloopcounter][my_TDAGL_counter]).lower() == 'none': ## CUSTOM START & END
                       key_HTRF.corr = 1;  # correct non-response
                    else:
                       key_HTRF.corr = 0;  # failed to respond (incorrectly)
                # store data for Trials_HTRF (TrialHandler)
                Trials_HTRF.addData('key_HTRF.keys',key_HTRF.keys)
                Trials_HTRF.addData('key_HTRF.corr', key_HTRF.corr)
                if key_HTRF.keys != None:  # we had a response
                    Trials_HTRF.addData('key_HTRF.rt', key_HTRF.rt)
                    Trials_HTRF.addData('key_HTRF.duration', key_HTRF.duration)
                try:
                    if image1_HTRF.tStopRefresh is not None:
                        duration_val = image1_HTRF.tStopRefresh - image1_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image1_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTRF).__name__,
                        trial_type='image1_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image1_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTRF.tStopRefresh is not None:
                        duration_val = image2_HTRF.tStopRefresh - image2_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image2_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTRF).__name__,
                        trial_type='image2_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image2_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTRF.tStopRefresh is not None:
                        duration_val = image3_HTRF.tStopRefresh - image3_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image3_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTRF).__name__,
                        trial_type='image3_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image3_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTRF.tStopRefresh is not None:
                        duration_val = image4_HTRF.tStopRefresh - image4_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image4_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTRF).__name__,
                        trial_type='image4_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image4_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTRF.tStopRefresh is not None:
                        duration_val = image5_HTRF.tStopRefresh - image5_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image5_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTRF).__name__,
                        trial_type='image5_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image5_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTRF.tStopRefresh is not None:
                        duration_val = image6_HTRF.tStopRefresh - image6_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image6_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTRF).__name__,
                        trial_type='image6_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image6_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTRF.tStopRefresh is not None:
                        duration_val = image7_HTRF.tStopRefresh - image7_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image7_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTRF).__name__,
                        trial_type='image7_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image7_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTRF.tStopRefresh is not None:
                        duration_val = image8_HTRF.tStopRefresh - image8_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image8_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTRF).__name__,
                        trial_type='image8_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image8_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTRF.tStopRefresh is not None:
                        duration_val = image9_HTRF.tStopRefresh - image9_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image9_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTRF).__name__,
                        trial_type='image9_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image9_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTRF.tStopRefresh is not None:
                        duration_val = image10_HTRF.tStopRefresh - image10_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image10_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTRF).__name__,
                        trial_type='image10_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image10_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTRF.tStopRefresh is not None:
                        duration_val = image11_HTRF.tStopRefresh - image11_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image11_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTRF).__name__,
                        trial_type='image11_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image11_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTRF.tStopRefresh is not None:
                        duration_val = image12_HTRF.tStopRefresh - image12_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image12_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTRF).__name__,
                        trial_type='image12_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image12_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTRF.tStopRefresh is not None:
                        duration_val = image13_HTRF.tStopRefresh - image13_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image13_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTRF).__name__,
                        trial_type='image13_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image13_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTRF.tStopRefresh is not None:
                        duration_val = image14_HTRF.tStopRefresh - image14_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image14_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTRF).__name__,
                        trial_type='image14_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image14_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTRF.tStopRefresh is not None:
                        duration_val = image15_HTRF.tStopRefresh - image15_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image15_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTRF).__name__,
                        trial_type='image15_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image15_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTRF.tStopRefresh is not None:
                        duration_val = image16_HTRF.tStopRefresh - image16_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image16_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTRF).__name__,
                        trial_type='image16_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image16_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTRF.tStopRefresh is not None:
                        duration_val = image17_HTRF.tStopRefresh - image17_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image17_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTRF).__name__,
                        trial_type='image17_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image17_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTRF.tStopRefresh is not None:
                        duration_val = image18_HTRF.tStopRefresh - image18_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image18_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTRF).__name__,
                        trial_type='image18_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image18_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTRF.tStopRefresh is not None:
                        duration_val = image19_HTRF.tStopRefresh - image19_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image19_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTRF).__name__,
                        trial_type='image19_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image19_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTRF.tStopRefresh is not None:
                        duration_val = image20_HTRF.tStopRefresh - image20_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image20_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTRF).__name__,
                        trial_type='image20_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image20_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTRF.tStopRefresh is not None:
                        duration_val = image21_HTRF.tStopRefresh - image21_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image21_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTRF).__name__,
                        trial_type='image21_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image21_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTRF.tStopRefresh is not None:
                        duration_val = image22_HTRF.tStopRefresh - image22_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image22_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTRF).__name__,
                        trial_type='image22_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image22_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTRF.tStopRefresh is not None:
                        duration_val = image23_HTRF.tStopRefresh - image23_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image23_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTRF).__name__,
                        trial_type='image23_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image23_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTRF.tStopRefresh is not None:
                        duration_val = image24_HTRF.tStopRefresh - image24_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image24_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTRF).__name__,
                        trial_type='image24_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image24_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTRF.tStopRefresh is not None:
                        duration_val = image_Cue_HTRF.tStopRefresh - image_Cue_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image_Cue_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTRF).__name__,
                        trial_type='image_Cue_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image_Cue_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTRF.tStopRefresh is not None:
                        duration_val = image_Tar_HTRF.tStopRefresh - image_Tar_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - image_Tar_HTRF.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTRF.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTRF).__name__,
                        trial_type='image_Tar_HTRF',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_image_Tar_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTRF.tStopRefresh is not None:
                        duration_val = key_HTRF.tStopRefresh - key_HTRF.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF.stopped'] - key_HTRF.tStartRefresh
                    if hasattr(key_HTRF, 'rt'):
                        rt_val = key_HTRF.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTRF" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTRF.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTRF).__name__,
                        trial_type='key_HTRF',
                        value=key_HTRF.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF.addData('bidsEvent_key_HTRF.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTRF.maxDurationReached:
                    routineTimer.addTime(-HTRF.maxDuration)
                elif HTRF.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                ## CUSTOM START
                my_TDAGL_counter = my_TDAGL_counter + 1
                ## CUSTOM END
                
            # completed 1.0 repeats of 'Trials_HTRF'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if Trials_HTRF.trialList in ([], [None], None):
                params = []
            else:
                params = Trials_HTRF.trialList[0].keys()
            # save data for this loop
            Trials_HTRF.saveAsExcel(filename + '.xlsx', sheetName='Trials_HTRF',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            Trials_HTRF.saveAsText(filename + 'Trials_HTRF.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
        #Custom-Comment: End TDA_GL
        elif my_seq_num[my_outerloopcounter]==4:
        #Custom-Comment: Start TDA_GR
        
            # set up handler to look after randomisation of conditions etc
            Trials_HTRU = data.TrialHandler2(
                name='Trials_HTRU',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TR.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(Trials_HTRU)  # add the loop to the experiment
            thisTrials_HTRU = Trials_HTRU.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRU.rgb)
            if thisTrials_HTRU != None:
                for paramName in thisTrials_HTRU:
                    globals()[paramName] = thisTrials_HTRU[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDAGR_counter = 0
            #--- CUSTOM END ---
            
            for thisTrials_HTRU in Trials_HTRU:
                currentLoop = Trials_HTRU
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRU.rgb)
                if thisTrials_HTRU != None:
                    for paramName in thisTrials_HTRU:
                        globals()[paramName] = thisTrials_HTRU[paramName]
                
                # --- Prepare to start Routine "HTRU" ---
                # create an object to store info about Routine HTRU
                HTRU = data.Routine(
                    name='HTRU',
                    components=[image1_HTRU, image2_HTRU, image3_HTRU, image4_HTRU, image5_HTRU, image6_HTRU, image7_HTRU, image8_HTRU, image9_HTRU, image10_HTRU, image11_HTRU, image12_HTRU, image13_HTRU, image14_HTRU, image15_HTRU, image16_HTRU, image17_HTRU, image18_HTRU, image19_HTRU, image20_HTRU, image21_HTRU, image22_HTRU, image23_HTRU, image24_HTRU, image_Frame_HTRU, image_Cue_HTRU, image_Cross_HTRU, image_Tar_HTRU, key_HTRU],
                )
                HTRU.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                    #--- CUSTOM START ---
                image_Tar_HTRU.setPos((my_coordTDA_GR[0, my_TDAGR_counter, my_outerloopcounter], my_coordTDA_GR[1, my_TDAGR_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTRU.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTRU
                key_HTRU.keys = []
                key_HTRU.rt = []
                _key_HTRU_allKeys = []
                # store start times for HTRU
                HTRU.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTRU.tStart = globalClock.getTime(format='float')
                HTRU.status = STARTED
                thisExp.addData('HTRU.started', HTRU.tStart)
                HTRU.maxDuration = None
                # keep track of which components have finished
                HTRUComponents = HTRU.components
                for thisComponent in HTRU.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTRU" ---
                # if trial has changed, end Routine now
                if isinstance(Trials_HTRU, data.TrialHandler2) and thisTrials_HTRU.thisN != Trials_HTRU.thisTrial.thisN:
                    continueRoutine = False
                HTRU.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTRU* updates
                    
                    # if image1_HTRU is starting this frame...
                    if image1_HTRU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTRU.frameNStart = frameN  # exact frame index
                        image1_HTRU.tStart = t  # local t and not account for scr refresh
                        image1_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTRU.started')
                        # update status
                        image1_HTRU.status = STARTED
                        image1_HTRU.setAutoDraw(True)
                    
                    # if image1_HTRU is active this frame...
                    if image1_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTRU is stopping this frame...
                    if image1_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTRU.tStop = t  # not accounting for scr refresh
                            image1_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTRU.stopped')
                            # update status
                            image1_HTRU.status = FINISHED
                            image1_HTRU.setAutoDraw(False)
                    
                    # *image2_HTRU* updates
                    
                    # if image2_HTRU is starting this frame...
                    if image2_HTRU.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTRU.frameNStart = frameN  # exact frame index
                        image2_HTRU.tStart = t  # local t and not account for scr refresh
                        image2_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTRU.started')
                        # update status
                        image2_HTRU.status = STARTED
                        image2_HTRU.setAutoDraw(True)
                    
                    # if image2_HTRU is active this frame...
                    if image2_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTRU is stopping this frame...
                    if image2_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTRU.tStop = t  # not accounting for scr refresh
                            image2_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTRU.stopped')
                            # update status
                            image2_HTRU.status = FINISHED
                            image2_HTRU.setAutoDraw(False)
                    
                    # *image3_HTRU* updates
                    
                    # if image3_HTRU is starting this frame...
                    if image3_HTRU.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTRU.frameNStart = frameN  # exact frame index
                        image3_HTRU.tStart = t  # local t and not account for scr refresh
                        image3_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTRU.started')
                        # update status
                        image3_HTRU.status = STARTED
                        image3_HTRU.setAutoDraw(True)
                    
                    # if image3_HTRU is active this frame...
                    if image3_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTRU is stopping this frame...
                    if image3_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTRU.tStop = t  # not accounting for scr refresh
                            image3_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTRU.stopped')
                            # update status
                            image3_HTRU.status = FINISHED
                            image3_HTRU.setAutoDraw(False)
                    
                    # *image4_HTRU* updates
                    
                    # if image4_HTRU is starting this frame...
                    if image4_HTRU.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTRU.frameNStart = frameN  # exact frame index
                        image4_HTRU.tStart = t  # local t and not account for scr refresh
                        image4_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTRU.started')
                        # update status
                        image4_HTRU.status = STARTED
                        image4_HTRU.setAutoDraw(True)
                    
                    # if image4_HTRU is active this frame...
                    if image4_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTRU is stopping this frame...
                    if image4_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTRU.tStop = t  # not accounting for scr refresh
                            image4_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTRU.stopped')
                            # update status
                            image4_HTRU.status = FINISHED
                            image4_HTRU.setAutoDraw(False)
                    
                    # *image5_HTRU* updates
                    
                    # if image5_HTRU is starting this frame...
                    if image5_HTRU.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTRU.frameNStart = frameN  # exact frame index
                        image5_HTRU.tStart = t  # local t and not account for scr refresh
                        image5_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTRU.started')
                        # update status
                        image5_HTRU.status = STARTED
                        image5_HTRU.setAutoDraw(True)
                    
                    # if image5_HTRU is active this frame...
                    if image5_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTRU is stopping this frame...
                    if image5_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTRU.tStop = t  # not accounting for scr refresh
                            image5_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTRU.stopped')
                            # update status
                            image5_HTRU.status = FINISHED
                            image5_HTRU.setAutoDraw(False)
                    
                    # *image6_HTRU* updates
                    
                    # if image6_HTRU is starting this frame...
                    if image6_HTRU.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTRU.frameNStart = frameN  # exact frame index
                        image6_HTRU.tStart = t  # local t and not account for scr refresh
                        image6_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTRU.started')
                        # update status
                        image6_HTRU.status = STARTED
                        image6_HTRU.setAutoDraw(True)
                    
                    # if image6_HTRU is active this frame...
                    if image6_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTRU is stopping this frame...
                    if image6_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTRU.tStop = t  # not accounting for scr refresh
                            image6_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTRU.stopped')
                            # update status
                            image6_HTRU.status = FINISHED
                            image6_HTRU.setAutoDraw(False)
                    
                    # *image7_HTRU* updates
                    
                    # if image7_HTRU is starting this frame...
                    if image7_HTRU.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTRU.frameNStart = frameN  # exact frame index
                        image7_HTRU.tStart = t  # local t and not account for scr refresh
                        image7_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTRU.started')
                        # update status
                        image7_HTRU.status = STARTED
                        image7_HTRU.setAutoDraw(True)
                    
                    # if image7_HTRU is active this frame...
                    if image7_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTRU is stopping this frame...
                    if image7_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTRU.tStop = t  # not accounting for scr refresh
                            image7_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTRU.stopped')
                            # update status
                            image7_HTRU.status = FINISHED
                            image7_HTRU.setAutoDraw(False)
                    
                    # *image8_HTRU* updates
                    
                    # if image8_HTRU is starting this frame...
                    if image8_HTRU.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTRU.frameNStart = frameN  # exact frame index
                        image8_HTRU.tStart = t  # local t and not account for scr refresh
                        image8_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTRU.started')
                        # update status
                        image8_HTRU.status = STARTED
                        image8_HTRU.setAutoDraw(True)
                    
                    # if image8_HTRU is active this frame...
                    if image8_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTRU is stopping this frame...
                    if image8_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTRU.tStop = t  # not accounting for scr refresh
                            image8_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTRU.stopped')
                            # update status
                            image8_HTRU.status = FINISHED
                            image8_HTRU.setAutoDraw(False)
                    
                    # *image9_HTRU* updates
                    
                    # if image9_HTRU is starting this frame...
                    if image9_HTRU.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTRU.frameNStart = frameN  # exact frame index
                        image9_HTRU.tStart = t  # local t and not account for scr refresh
                        image9_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTRU.started')
                        # update status
                        image9_HTRU.status = STARTED
                        image9_HTRU.setAutoDraw(True)
                    
                    # if image9_HTRU is active this frame...
                    if image9_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTRU is stopping this frame...
                    if image9_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTRU.tStop = t  # not accounting for scr refresh
                            image9_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTRU.stopped')
                            # update status
                            image9_HTRU.status = FINISHED
                            image9_HTRU.setAutoDraw(False)
                    
                    # *image10_HTRU* updates
                    
                    # if image10_HTRU is starting this frame...
                    if image10_HTRU.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTRU.frameNStart = frameN  # exact frame index
                        image10_HTRU.tStart = t  # local t and not account for scr refresh
                        image10_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTRU.started')
                        # update status
                        image10_HTRU.status = STARTED
                        image10_HTRU.setAutoDraw(True)
                    
                    # if image10_HTRU is active this frame...
                    if image10_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTRU is stopping this frame...
                    if image10_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTRU.tStop = t  # not accounting for scr refresh
                            image10_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTRU.stopped')
                            # update status
                            image10_HTRU.status = FINISHED
                            image10_HTRU.setAutoDraw(False)
                    
                    # *image11_HTRU* updates
                    
                    # if image11_HTRU is starting this frame...
                    if image11_HTRU.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTRU.frameNStart = frameN  # exact frame index
                        image11_HTRU.tStart = t  # local t and not account for scr refresh
                        image11_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTRU.started')
                        # update status
                        image11_HTRU.status = STARTED
                        image11_HTRU.setAutoDraw(True)
                    
                    # if image11_HTRU is active this frame...
                    if image11_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTRU is stopping this frame...
                    if image11_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTRU.tStop = t  # not accounting for scr refresh
                            image11_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTRU.stopped')
                            # update status
                            image11_HTRU.status = FINISHED
                            image11_HTRU.setAutoDraw(False)
                    
                    # *image12_HTRU* updates
                    
                    # if image12_HTRU is starting this frame...
                    if image12_HTRU.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTRU.frameNStart = frameN  # exact frame index
                        image12_HTRU.tStart = t  # local t and not account for scr refresh
                        image12_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTRU.started')
                        # update status
                        image12_HTRU.status = STARTED
                        image12_HTRU.setAutoDraw(True)
                    
                    # if image12_HTRU is active this frame...
                    if image12_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTRU is stopping this frame...
                    if image12_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTRU.tStop = t  # not accounting for scr refresh
                            image12_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTRU.stopped')
                            # update status
                            image12_HTRU.status = FINISHED
                            image12_HTRU.setAutoDraw(False)
                    
                    # *image13_HTRU* updates
                    
                    # if image13_HTRU is starting this frame...
                    if image13_HTRU.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTRU.frameNStart = frameN  # exact frame index
                        image13_HTRU.tStart = t  # local t and not account for scr refresh
                        image13_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTRU.started')
                        # update status
                        image13_HTRU.status = STARTED
                        image13_HTRU.setAutoDraw(True)
                    
                    # if image13_HTRU is active this frame...
                    if image13_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTRU is stopping this frame...
                    if image13_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTRU.tStop = t  # not accounting for scr refresh
                            image13_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTRU.stopped')
                            # update status
                            image13_HTRU.status = FINISHED
                            image13_HTRU.setAutoDraw(False)
                    
                    # *image14_HTRU* updates
                    
                    # if image14_HTRU is starting this frame...
                    if image14_HTRU.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTRU.frameNStart = frameN  # exact frame index
                        image14_HTRU.tStart = t  # local t and not account for scr refresh
                        image14_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTRU.started')
                        # update status
                        image14_HTRU.status = STARTED
                        image14_HTRU.setAutoDraw(True)
                    
                    # if image14_HTRU is active this frame...
                    if image14_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTRU is stopping this frame...
                    if image14_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTRU.tStop = t  # not accounting for scr refresh
                            image14_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTRU.stopped')
                            # update status
                            image14_HTRU.status = FINISHED
                            image14_HTRU.setAutoDraw(False)
                    
                    # *image15_HTRU* updates
                    
                    # if image15_HTRU is starting this frame...
                    if image15_HTRU.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTRU.frameNStart = frameN  # exact frame index
                        image15_HTRU.tStart = t  # local t and not account for scr refresh
                        image15_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTRU.started')
                        # update status
                        image15_HTRU.status = STARTED
                        image15_HTRU.setAutoDraw(True)
                    
                    # if image15_HTRU is active this frame...
                    if image15_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTRU is stopping this frame...
                    if image15_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTRU.tStop = t  # not accounting for scr refresh
                            image15_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTRU.stopped')
                            # update status
                            image15_HTRU.status = FINISHED
                            image15_HTRU.setAutoDraw(False)
                    
                    # *image16_HTRU* updates
                    
                    # if image16_HTRU is starting this frame...
                    if image16_HTRU.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTRU.frameNStart = frameN  # exact frame index
                        image16_HTRU.tStart = t  # local t and not account for scr refresh
                        image16_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTRU.started')
                        # update status
                        image16_HTRU.status = STARTED
                        image16_HTRU.setAutoDraw(True)
                    
                    # if image16_HTRU is active this frame...
                    if image16_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTRU is stopping this frame...
                    if image16_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTRU.tStop = t  # not accounting for scr refresh
                            image16_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTRU.stopped')
                            # update status
                            image16_HTRU.status = FINISHED
                            image16_HTRU.setAutoDraw(False)
                    
                    # *image17_HTRU* updates
                    
                    # if image17_HTRU is starting this frame...
                    if image17_HTRU.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTRU.frameNStart = frameN  # exact frame index
                        image17_HTRU.tStart = t  # local t and not account for scr refresh
                        image17_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTRU.started')
                        # update status
                        image17_HTRU.status = STARTED
                        image17_HTRU.setAutoDraw(True)
                    
                    # if image17_HTRU is active this frame...
                    if image17_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTRU is stopping this frame...
                    if image17_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTRU.tStop = t  # not accounting for scr refresh
                            image17_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTRU.stopped')
                            # update status
                            image17_HTRU.status = FINISHED
                            image17_HTRU.setAutoDraw(False)
                    
                    # *image18_HTRU* updates
                    
                    # if image18_HTRU is starting this frame...
                    if image18_HTRU.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTRU.frameNStart = frameN  # exact frame index
                        image18_HTRU.tStart = t  # local t and not account for scr refresh
                        image18_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTRU.started')
                        # update status
                        image18_HTRU.status = STARTED
                        image18_HTRU.setAutoDraw(True)
                    
                    # if image18_HTRU is active this frame...
                    if image18_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTRU is stopping this frame...
                    if image18_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTRU.tStop = t  # not accounting for scr refresh
                            image18_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTRU.stopped')
                            # update status
                            image18_HTRU.status = FINISHED
                            image18_HTRU.setAutoDraw(False)
                    
                    # *image19_HTRU* updates
                    
                    # if image19_HTRU is starting this frame...
                    if image19_HTRU.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTRU.frameNStart = frameN  # exact frame index
                        image19_HTRU.tStart = t  # local t and not account for scr refresh
                        image19_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTRU.started')
                        # update status
                        image19_HTRU.status = STARTED
                        image19_HTRU.setAutoDraw(True)
                    
                    # if image19_HTRU is active this frame...
                    if image19_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTRU is stopping this frame...
                    if image19_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTRU.tStop = t  # not accounting for scr refresh
                            image19_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTRU.stopped')
                            # update status
                            image19_HTRU.status = FINISHED
                            image19_HTRU.setAutoDraw(False)
                    
                    # *image20_HTRU* updates
                    
                    # if image20_HTRU is starting this frame...
                    if image20_HTRU.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTRU.frameNStart = frameN  # exact frame index
                        image20_HTRU.tStart = t  # local t and not account for scr refresh
                        image20_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTRU.started')
                        # update status
                        image20_HTRU.status = STARTED
                        image20_HTRU.setAutoDraw(True)
                    
                    # if image20_HTRU is active this frame...
                    if image20_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTRU is stopping this frame...
                    if image20_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTRU.tStop = t  # not accounting for scr refresh
                            image20_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTRU.stopped')
                            # update status
                            image20_HTRU.status = FINISHED
                            image20_HTRU.setAutoDraw(False)
                    
                    # *image21_HTRU* updates
                    
                    # if image21_HTRU is starting this frame...
                    if image21_HTRU.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTRU.frameNStart = frameN  # exact frame index
                        image21_HTRU.tStart = t  # local t and not account for scr refresh
                        image21_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTRU.started')
                        # update status
                        image21_HTRU.status = STARTED
                        image21_HTRU.setAutoDraw(True)
                    
                    # if image21_HTRU is active this frame...
                    if image21_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTRU is stopping this frame...
                    if image21_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTRU.tStop = t  # not accounting for scr refresh
                            image21_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTRU.stopped')
                            # update status
                            image21_HTRU.status = FINISHED
                            image21_HTRU.setAutoDraw(False)
                    
                    # *image22_HTRU* updates
                    
                    # if image22_HTRU is starting this frame...
                    if image22_HTRU.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTRU.frameNStart = frameN  # exact frame index
                        image22_HTRU.tStart = t  # local t and not account for scr refresh
                        image22_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTRU.started')
                        # update status
                        image22_HTRU.status = STARTED
                        image22_HTRU.setAutoDraw(True)
                    
                    # if image22_HTRU is active this frame...
                    if image22_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTRU is stopping this frame...
                    if image22_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTRU.tStop = t  # not accounting for scr refresh
                            image22_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTRU.stopped')
                            # update status
                            image22_HTRU.status = FINISHED
                            image22_HTRU.setAutoDraw(False)
                    
                    # *image23_HTRU* updates
                    
                    # if image23_HTRU is starting this frame...
                    if image23_HTRU.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTRU.frameNStart = frameN  # exact frame index
                        image23_HTRU.tStart = t  # local t and not account for scr refresh
                        image23_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTRU.started')
                        # update status
                        image23_HTRU.status = STARTED
                        image23_HTRU.setAutoDraw(True)
                    
                    # if image23_HTRU is active this frame...
                    if image23_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTRU is stopping this frame...
                    if image23_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTRU.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTRU.tStop = t  # not accounting for scr refresh
                            image23_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTRU.stopped')
                            # update status
                            image23_HTRU.status = FINISHED
                            image23_HTRU.setAutoDraw(False)
                    
                    # *image24_HTRU* updates
                    
                    # if image24_HTRU is starting this frame...
                    if image24_HTRU.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTRU.frameNStart = frameN  # exact frame index
                        image24_HTRU.tStart = t  # local t and not account for scr refresh
                        image24_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTRU.started')
                        # update status
                        image24_HTRU.status = STARTED
                        image24_HTRU.setAutoDraw(True)
                    
                    # if image24_HTRU is active this frame...
                    if image24_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTRU is stopping this frame...
                    if image24_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTRU.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTRU.tStop = t  # not accounting for scr refresh
                            image24_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTRU.stopped')
                            # update status
                            image24_HTRU.status = FINISHED
                            image24_HTRU.setAutoDraw(False)
                    
                    # *image_Frame_HTRU* updates
                    
                    # if image_Frame_HTRU is starting this frame...
                    if image_Frame_HTRU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTRU.frameNStart = frameN  # exact frame index
                        image_Frame_HTRU.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTRU.started')
                        # update status
                        image_Frame_HTRU.status = STARTED
                        image_Frame_HTRU.setAutoDraw(True)
                    
                    # if image_Frame_HTRU is active this frame...
                    if image_Frame_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTRU is stopping this frame...
                    if image_Frame_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTRU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTRU.tStop = t  # not accounting for scr refresh
                            image_Frame_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTRU.stopped')
                            # update status
                            image_Frame_HTRU.status = FINISHED
                            image_Frame_HTRU.setAutoDraw(False)
                    
                    # *image_Cue_HTRU* updates
                    
                    # if image_Cue_HTRU is starting this frame...
                    if image_Cue_HTRU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTRU.frameNStart = frameN  # exact frame index
                        image_Cue_HTRU.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTRU.started')
                        # update status
                        image_Cue_HTRU.status = STARTED
                        image_Cue_HTRU.setAutoDraw(True)
                    
                    # if image_Cue_HTRU is active this frame...
                    if image_Cue_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTRU is stopping this frame...
                    if image_Cue_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTRU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTRU.tStop = t  # not accounting for scr refresh
                            image_Cue_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTRU.stopped')
                            # update status
                            image_Cue_HTRU.status = FINISHED
                            image_Cue_HTRU.setAutoDraw(False)
                    
                    # *image_Cross_HTRU* updates
                    
                    # if image_Cross_HTRU is starting this frame...
                    if image_Cross_HTRU.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTRU.frameNStart = frameN  # exact frame index
                        image_Cross_HTRU.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTRU.started')
                        # update status
                        image_Cross_HTRU.status = STARTED
                        image_Cross_HTRU.setAutoDraw(True)
                    
                    # if image_Cross_HTRU is active this frame...
                    if image_Cross_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTRU is stopping this frame...
                    if image_Cross_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTRU.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTRU.tStop = t  # not accounting for scr refresh
                            image_Cross_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTRU.stopped')
                            # update status
                            image_Cross_HTRU.status = FINISHED
                            image_Cross_HTRU.setAutoDraw(False)
                    
                    # *image_Tar_HTRU* updates
                    
                    # if image_Tar_HTRU is starting this frame...
                    if image_Tar_HTRU.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTRU.frameNStart = frameN  # exact frame index
                        image_Tar_HTRU.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTRU.started')
                        # update status
                        image_Tar_HTRU.status = STARTED
                        image_Tar_HTRU.setAutoDraw(True)
                    
                    # if image_Tar_HTRU is active this frame...
                    if image_Tar_HTRU.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTRU is stopping this frame...
                    if image_Tar_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTRU.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END 
                            # keep track of stop time/frame for later
                            image_Tar_HTRU.tStop = t  # not accounting for scr refresh
                            image_Tar_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTRU.stopped')
                            # update status
                            image_Tar_HTRU.status = FINISHED
                            image_Tar_HTRU.setAutoDraw(False)
                    
                    # *key_HTRU* updates
                    waitOnFlip = False
                    
                    # if key_HTRU is starting this frame...
                    if key_HTRU.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTRU.frameNStart = frameN  # exact frame index
                        key_HTRU.tStart = t  # local t and not account for scr refresh
                        key_HTRU.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTRU, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTRU.started')
                        # update status
                        key_HTRU.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTRU.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTRU.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTRU is stopping this frame...
                    if key_HTRU.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTRU.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTRU.tStop = t  # not accounting for scr refresh
                            key_HTRU.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTRU.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTRU.stopped')
                            # update status
                            key_HTRU.status = FINISHED
                            key_HTRU.status = FINISHED
                    if key_HTRU.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTRU.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTRU_allKeys.extend(theseKeys)
                        if len(_key_HTRU_allKeys):
                            key_HTRU.keys = _key_HTRU_allKeys[0].name  # just the first key pressed
                            key_HTRU.rt = _key_HTRU_allKeys[0].rt
                            key_HTRU.duration = _key_HTRU_allKeys[0].duration
                            # was this correct?
                            ### CUSTOM START ###
                            #if (key_HTRU.keys == str(corr_resp)) or (key_HTRU.keys == corr_resp):
                            if (key_HTRU.keys == str(my_corrResp_GR[my_outerloopcounter][my_TDAGR_counter])) or (key_HTRU.keys == my_corrResp_GR[my_outerloopcounter][my_TDAGR_counter]):
                                key_HTRU.corr = 1
                            else:
                                key_HTRU.corr = 0
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTRU.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTRU.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END
                
                # --- Ending Routine "HTRU" ---
                for thisComponent in HTRU.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTRU
                HTRU.tStop = globalClock.getTime(format='float')
                HTRU.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTRU.stopped', HTRU.tStop)
                # check responses
                if key_HTRU.keys in ['', [], None]:  # No response was made
                    key_HTRU.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_GR[my_outerloopcounter][my_TDAGR_counter]).lower() == 'none': # CUSTOM START & END
                       key_HTRU.corr = 1;  # correct non-response
                    else:
                       key_HTRU.corr = 0;  # failed to respond (incorrectly)
                # store data for Trials_HTRU (TrialHandler)
                Trials_HTRU.addData('key_HTRU.keys',key_HTRU.keys)
                Trials_HTRU.addData('key_HTRU.corr', key_HTRU.corr)
                if key_HTRU.keys != None:  # we had a response
                    Trials_HTRU.addData('key_HTRU.rt', key_HTRU.rt)
                    Trials_HTRU.addData('key_HTRU.duration', key_HTRU.duration)
                try:
                    if image1_HTRU.tStopRefresh is not None:
                        duration_val = image1_HTRU.tStopRefresh - image1_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image1_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTRU).__name__,
                        trial_type='image1_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image1_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTRU.tStopRefresh is not None:
                        duration_val = image2_HTRU.tStopRefresh - image2_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image2_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTRU).__name__,
                        trial_type='image2_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image2_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTRU.tStopRefresh is not None:
                        duration_val = image3_HTRU.tStopRefresh - image3_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image3_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTRU).__name__,
                        trial_type='image3_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image3_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTRU.tStopRefresh is not None:
                        duration_val = image4_HTRU.tStopRefresh - image4_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image4_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTRU).__name__,
                        trial_type='image4_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image4_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTRU.tStopRefresh is not None:
                        duration_val = image5_HTRU.tStopRefresh - image5_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image5_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTRU).__name__,
                        trial_type='image5_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image5_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTRU.tStopRefresh is not None:
                        duration_val = image6_HTRU.tStopRefresh - image6_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image6_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTRU).__name__,
                        trial_type='image6_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image6_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTRU.tStopRefresh is not None:
                        duration_val = image7_HTRU.tStopRefresh - image7_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image7_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTRU).__name__,
                        trial_type='image7_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image7_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTRU.tStopRefresh is not None:
                        duration_val = image8_HTRU.tStopRefresh - image8_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image8_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTRU).__name__,
                        trial_type='image8_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image8_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTRU.tStopRefresh is not None:
                        duration_val = image9_HTRU.tStopRefresh - image9_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image9_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTRU).__name__,
                        trial_type='image9_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image9_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTRU.tStopRefresh is not None:
                        duration_val = image10_HTRU.tStopRefresh - image10_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image10_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTRU).__name__,
                        trial_type='image10_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image10_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTRU.tStopRefresh is not None:
                        duration_val = image11_HTRU.tStopRefresh - image11_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image11_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTRU).__name__,
                        trial_type='image11_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image11_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTRU.tStopRefresh is not None:
                        duration_val = image12_HTRU.tStopRefresh - image12_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image12_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTRU).__name__,
                        trial_type='image12_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image12_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTRU.tStopRefresh is not None:
                        duration_val = image13_HTRU.tStopRefresh - image13_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image13_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTRU).__name__,
                        trial_type='image13_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image13_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTRU.tStopRefresh is not None:
                        duration_val = image14_HTRU.tStopRefresh - image14_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image14_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTRU).__name__,
                        trial_type='image14_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image14_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTRU.tStopRefresh is not None:
                        duration_val = image15_HTRU.tStopRefresh - image15_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image15_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTRU).__name__,
                        trial_type='image15_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image15_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTRU.tStopRefresh is not None:
                        duration_val = image16_HTRU.tStopRefresh - image16_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image16_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTRU).__name__,
                        trial_type='image16_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image16_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTRU.tStopRefresh is not None:
                        duration_val = image17_HTRU.tStopRefresh - image17_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image17_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTRU).__name__,
                        trial_type='image17_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image17_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTRU.tStopRefresh is not None:
                        duration_val = image18_HTRU.tStopRefresh - image18_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image18_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTRU).__name__,
                        trial_type='image18_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image18_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTRU.tStopRefresh is not None:
                        duration_val = image19_HTRU.tStopRefresh - image19_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image19_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTRU).__name__,
                        trial_type='image19_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image19_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTRU.tStopRefresh is not None:
                        duration_val = image20_HTRU.tStopRefresh - image20_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image20_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTRU).__name__,
                        trial_type='image20_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image20_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTRU.tStopRefresh is not None:
                        duration_val = image21_HTRU.tStopRefresh - image21_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image21_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTRU).__name__,
                        trial_type='image21_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image21_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTRU.tStopRefresh is not None:
                        duration_val = image22_HTRU.tStopRefresh - image22_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image22_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTRU).__name__,
                        trial_type='image22_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image22_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTRU.tStopRefresh is not None:
                        duration_val = image23_HTRU.tStopRefresh - image23_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image23_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTRU).__name__,
                        trial_type='image23_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image23_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTRU.tStopRefresh is not None:
                        duration_val = image24_HTRU.tStopRefresh - image24_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image24_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTRU).__name__,
                        trial_type='image24_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image24_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTRU.tStopRefresh is not None:
                        duration_val = image_Cue_HTRU.tStopRefresh - image_Cue_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image_Cue_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTRU).__name__,
                        trial_type='image_Cue_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image_Cue_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTRU.tStopRefresh is not None:
                        duration_val = image_Tar_HTRU.tStopRefresh - image_Tar_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - image_Tar_HTRU.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTRU.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTRU).__name__,
                        trial_type='image_Tar_HTRU',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_image_Tar_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTRU.tStopRefresh is not None:
                        duration_val = key_HTRU.tStopRefresh - key_HTRU.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRU.stopped'] - key_HTRU.tStartRefresh
                    if hasattr(key_HTRU, 'rt'):
                        rt_val = key_HTRU.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTRU" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTRU.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTRU).__name__,
                        trial_type='key_HTRU',
                        value=key_HTRU.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRU.addData('bidsEvent_key_HTRU.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTRU.maxDurationReached:
                    routineTimer.addTime(-HTRU.maxDuration)
                elif HTRU.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                ## CUSTOM START 
                my_TDAGR_counter = my_TDAGR_counter + 1 
                ## CUSTOM END 
                
            # completed 1.0 repeats of 'Trials_HTRU'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if Trials_HTRU.trialList in ([], [None], None):
                params = []
            else:
                params = Trials_HTRU.trialList[0].keys()
            # save data for this loop
            Trials_HTRU.saveAsExcel(filename + '.xlsx', sheetName='Trials_HTRU',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            Trials_HTRU.saveAsText(filename + 'Trials_HTRU.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
        #Custom-Comment: End TDA_GR
        elif my_seq_num[my_outerloopcounter]==5:
        #Custom-Comment: Start TDA_L_OD
        
            # set up handler to look after randomisation of conditions etc
            TRials_HTLF_OD = data.TrialHandler2(
                name='TRials_HTLF_OD',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TL_ND.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(TRials_HTLF_OD)  # add the loop to the experiment
            thisTRials_HTLF_OD = TRials_HTLF_OD.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTRials_HTLF_OD.rgb)
            if thisTRials_HTLF_OD != None:
                for paramName in thisTRials_HTLF_OD:
                    globals()[paramName] = thisTRials_HTLF_OD[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDA_L_OD_counter = 0
            #--- CUSTOM END ---
            
            for thisTRials_HTLF_OD in TRials_HTLF_OD:
                currentLoop = TRials_HTLF_OD
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTRials_HTLF_OD.rgb)
                if thisTRials_HTLF_OD != None:
                    for paramName in thisTRials_HTLF_OD:
                        globals()[paramName] = thisTRials_HTLF_OD[paramName]
                
                # --- Prepare to start Routine "HTLF_OD" ---
                # create an object to store info about Routine HTLF_OD
                HTLF_OD = data.Routine(
                    name='HTLF_OD',
                    components=[image1_HTLF_OD, image2_HTLF_OD, image3_HTLF_OD, image4_HTLF_OD, image5_HTLF_OD, image6_HTLF_OD, image7_HTLF_OD, image8_HTLF_OD, image9_HTLF_OD, image10_HTLF_OD, image11_HTLF_OD, image12_HTLF_OD, image13_HTLF_OD, image14_HTLF_OD, image15_HTLF_OD, image16_HTLF_OD, image17_HTLF_OD, image18_HTLF_OD, image19_HTLF_OD, image20_HTLF_OD, image21_HTLF_OD, image22_HTLF_OD, image23_HTLF_OD, image24_HTLF_OD, image_Frame_HTLF_OD, image_Cue_HTLF_OD, image_Cross_HTLF_OD, image_Tar_HTLF_OD, key_HTLF_OD],
                )
                HTLF_OD.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                #--- CUSTOM START ---
                image_Tar_HTLF_OD.setPos((my_coordTDA_L_OD[0, my_TDA_L_OD_counter, my_outerloopcounter], my_coordTDA_L_OD[1, my_TDA_L_OD_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTLF_OD.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTLF_OD
                key_HTLF_OD.keys = []
                key_HTLF_OD.rt = []
                _key_HTLF_OD_allKeys = []
                # store start times for HTLF_OD
                HTLF_OD.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTLF_OD.tStart = globalClock.getTime(format='float')
                HTLF_OD.status = STARTED
                thisExp.addData('HTLF_OD.started', HTLF_OD.tStart)
                HTLF_OD.maxDuration = None
                # keep track of which components have finished
                HTLF_ODComponents = HTLF_OD.components
                for thisComponent in HTLF_OD.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTLF_OD" ---
                # if trial has changed, end Routine now
                if isinstance(TRials_HTLF_OD, data.TrialHandler2) and thisTRials_HTLF_OD.thisN != TRials_HTLF_OD.thisTrial.thisN:
                    continueRoutine = False
                HTLF_OD.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTLF_OD* updates
                    
                    # if image1_HTLF_OD is starting this frame...
                    if image1_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTLF_OD.frameNStart = frameN  # exact frame index
                        image1_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image1_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTLF_OD.started')
                        # update status
                        image1_HTLF_OD.status = STARTED
                        image1_HTLF_OD.setAutoDraw(True)
                    
                    # if image1_HTLF_OD is active this frame...
                    if image1_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTLF_OD is stopping this frame...
                    if image1_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image1_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTLF_OD.stopped')
                            # update status
                            image1_HTLF_OD.status = FINISHED
                            image1_HTLF_OD.setAutoDraw(False)
                    
                    # *image2_HTLF_OD* updates
                    
                    # if image2_HTLF_OD is starting this frame...
                    if image2_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTLF_OD.frameNStart = frameN  # exact frame index
                        image2_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image2_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTLF_OD.started')
                        # update status
                        image2_HTLF_OD.status = STARTED
                        image2_HTLF_OD.setAutoDraw(True)
                    
                    # if image2_HTLF_OD is active this frame...
                    if image2_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTLF_OD is stopping this frame...
                    if image2_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image2_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTLF_OD.stopped')
                            # update status
                            image2_HTLF_OD.status = FINISHED
                            image2_HTLF_OD.setAutoDraw(False)
                    
                    # *image3_HTLF_OD* updates
                    
                    # if image3_HTLF_OD is starting this frame...
                    if image3_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTLF_OD.frameNStart = frameN  # exact frame index
                        image3_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image3_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTLF_OD.started')
                        # update status
                        image3_HTLF_OD.status = STARTED
                        image3_HTLF_OD.setAutoDraw(True)
                    
                    # if image3_HTLF_OD is active this frame...
                    if image3_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTLF_OD is stopping this frame...
                    if image3_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image3_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTLF_OD.stopped')
                            # update status
                            image3_HTLF_OD.status = FINISHED
                            image3_HTLF_OD.setAutoDraw(False)
                    
                    # *image4_HTLF_OD* updates
                    
                    # if image4_HTLF_OD is starting this frame...
                    if image4_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTLF_OD.frameNStart = frameN  # exact frame index
                        image4_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image4_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTLF_OD.started')
                        # update status
                        image4_HTLF_OD.status = STARTED
                        image4_HTLF_OD.setAutoDraw(True)
                    
                    # if image4_HTLF_OD is active this frame...
                    if image4_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTLF_OD is stopping this frame...
                    if image4_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image4_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTLF_OD.stopped')
                            # update status
                            image4_HTLF_OD.status = FINISHED
                            image4_HTLF_OD.setAutoDraw(False)
                    
                    # *image5_HTLF_OD* updates
                    
                    # if image5_HTLF_OD is starting this frame...
                    if image5_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTLF_OD.frameNStart = frameN  # exact frame index
                        image5_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image5_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTLF_OD.started')
                        # update status
                        image5_HTLF_OD.status = STARTED
                        image5_HTLF_OD.setAutoDraw(True)
                    
                    # if image5_HTLF_OD is active this frame...
                    if image5_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTLF_OD is stopping this frame...
                    if image5_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image5_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTLF_OD.stopped')
                            # update status
                            image5_HTLF_OD.status = FINISHED
                            image5_HTLF_OD.setAutoDraw(False)
                    
                    # *image6_HTLF_OD* updates
                    
                    # if image6_HTLF_OD is starting this frame...
                    if image6_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTLF_OD.frameNStart = frameN  # exact frame index
                        image6_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image6_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTLF_OD.started')
                        # update status
                        image6_HTLF_OD.status = STARTED
                        image6_HTLF_OD.setAutoDraw(True)
                    
                    # if image6_HTLF_OD is active this frame...
                    if image6_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTLF_OD is stopping this frame...
                    if image6_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image6_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTLF_OD.stopped')
                            # update status
                            image6_HTLF_OD.status = FINISHED
                            image6_HTLF_OD.setAutoDraw(False)
                    
                    # *image7_HTLF_OD* updates
                    
                    # if image7_HTLF_OD is starting this frame...
                    if image7_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTLF_OD.frameNStart = frameN  # exact frame index
                        image7_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image7_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTLF_OD.started')
                        # update status
                        image7_HTLF_OD.status = STARTED
                        image7_HTLF_OD.setAutoDraw(True)
                    
                    # if image7_HTLF_OD is active this frame...
                    if image7_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTLF_OD is stopping this frame...
                    if image7_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image7_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTLF_OD.stopped')
                            # update status
                            image7_HTLF_OD.status = FINISHED
                            image7_HTLF_OD.setAutoDraw(False)
                    
                    # *image8_HTLF_OD* updates
                    
                    # if image8_HTLF_OD is starting this frame...
                    if image8_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTLF_OD.frameNStart = frameN  # exact frame index
                        image8_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image8_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTLF_OD.started')
                        # update status
                        image8_HTLF_OD.status = STARTED
                        image8_HTLF_OD.setAutoDraw(True)
                    
                    # if image8_HTLF_OD is active this frame...
                    if image8_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTLF_OD is stopping this frame...
                    if image8_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image8_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTLF_OD.stopped')
                            # update status
                            image8_HTLF_OD.status = FINISHED
                            image8_HTLF_OD.setAutoDraw(False)
                    
                    # *image9_HTLF_OD* updates
                    
                    # if image9_HTLF_OD is starting this frame...
                    if image9_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTLF_OD.frameNStart = frameN  # exact frame index
                        image9_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image9_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTLF_OD.started')
                        # update status
                        image9_HTLF_OD.status = STARTED
                        image9_HTLF_OD.setAutoDraw(True)
                    
                    # if image9_HTLF_OD is active this frame...
                    if image9_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTLF_OD is stopping this frame...
                    if image9_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image9_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTLF_OD.stopped')
                            # update status
                            image9_HTLF_OD.status = FINISHED
                            image9_HTLF_OD.setAutoDraw(False)
                    
                    # *image10_HTLF_OD* updates
                    
                    # if image10_HTLF_OD is starting this frame...
                    if image10_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTLF_OD.frameNStart = frameN  # exact frame index
                        image10_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image10_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTLF_OD.started')
                        # update status
                        image10_HTLF_OD.status = STARTED
                        image10_HTLF_OD.setAutoDraw(True)
                    
                    # if image10_HTLF_OD is active this frame...
                    if image10_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTLF_OD is stopping this frame...
                    if image10_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image10_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTLF_OD.stopped')
                            # update status
                            image10_HTLF_OD.status = FINISHED
                            image10_HTLF_OD.setAutoDraw(False)
                    
                    # *image11_HTLF_OD* updates
                    
                    # if image11_HTLF_OD is starting this frame...
                    if image11_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTLF_OD.frameNStart = frameN  # exact frame index
                        image11_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image11_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTLF_OD.started')
                        # update status
                        image11_HTLF_OD.status = STARTED
                        image11_HTLF_OD.setAutoDraw(True)
                    
                    # if image11_HTLF_OD is active this frame...
                    if image11_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTLF_OD is stopping this frame...
                    if image11_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image11_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTLF_OD.stopped')
                            # update status
                            image11_HTLF_OD.status = FINISHED
                            image11_HTLF_OD.setAutoDraw(False)
                    
                    # *image12_HTLF_OD* updates
                    
                    # if image12_HTLF_OD is starting this frame...
                    if image12_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTLF_OD.frameNStart = frameN  # exact frame index
                        image12_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image12_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTLF_OD.started')
                        # update status
                        image12_HTLF_OD.status = STARTED
                        image12_HTLF_OD.setAutoDraw(True)
                    
                    # if image12_HTLF_OD is active this frame...
                    if image12_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTLF_OD is stopping this frame...
                    if image12_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image12_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTLF_OD.stopped')
                            # update status
                            image12_HTLF_OD.status = FINISHED
                            image12_HTLF_OD.setAutoDraw(False)
                    
                    # *image13_HTLF_OD* updates
                    
                    # if image13_HTLF_OD is starting this frame...
                    if image13_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTLF_OD.frameNStart = frameN  # exact frame index
                        image13_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image13_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTLF_OD.started')
                        # update status
                        image13_HTLF_OD.status = STARTED
                        image13_HTLF_OD.setAutoDraw(True)
                    
                    # if image13_HTLF_OD is active this frame...
                    if image13_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTLF_OD is stopping this frame...
                    if image13_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image13_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTLF_OD.stopped')
                            # update status
                            image13_HTLF_OD.status = FINISHED
                            image13_HTLF_OD.setAutoDraw(False)
                    
                    # *image14_HTLF_OD* updates
                    
                    # if image14_HTLF_OD is starting this frame...
                    if image14_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTLF_OD.frameNStart = frameN  # exact frame index
                        image14_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image14_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTLF_OD.started')
                        # update status
                        image14_HTLF_OD.status = STARTED
                        image14_HTLF_OD.setAutoDraw(True)
                    
                    # if image14_HTLF_OD is active this frame...
                    if image14_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTLF_OD is stopping this frame...
                    if image14_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image14_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTLF_OD.stopped')
                            # update status
                            image14_HTLF_OD.status = FINISHED
                            image14_HTLF_OD.setAutoDraw(False)
                    
                    # *image15_HTLF_OD* updates
                    
                    # if image15_HTLF_OD is starting this frame...
                    if image15_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTLF_OD.frameNStart = frameN  # exact frame index
                        image15_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image15_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTLF_OD.started')
                        # update status
                        image15_HTLF_OD.status = STARTED
                        image15_HTLF_OD.setAutoDraw(True)
                    
                    # if image15_HTLF_OD is active this frame...
                    if image15_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTLF_OD is stopping this frame...
                    if image15_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image15_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTLF_OD.stopped')
                            # update status
                            image15_HTLF_OD.status = FINISHED
                            image15_HTLF_OD.setAutoDraw(False)
                    
                    # *image16_HTLF_OD* updates
                    
                    # if image16_HTLF_OD is starting this frame...
                    if image16_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTLF_OD.frameNStart = frameN  # exact frame index
                        image16_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image16_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTLF_OD.started')
                        # update status
                        image16_HTLF_OD.status = STARTED
                        image16_HTLF_OD.setAutoDraw(True)
                    
                    # if image16_HTLF_OD is active this frame...
                    if image16_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTLF_OD is stopping this frame...
                    if image16_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image16_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTLF_OD.stopped')
                            # update status
                            image16_HTLF_OD.status = FINISHED
                            image16_HTLF_OD.setAutoDraw(False)
                    
                    # *image17_HTLF_OD* updates
                    
                    # if image17_HTLF_OD is starting this frame...
                    if image17_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTLF_OD.frameNStart = frameN  # exact frame index
                        image17_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image17_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTLF_OD.started')
                        # update status
                        image17_HTLF_OD.status = STARTED
                        image17_HTLF_OD.setAutoDraw(True)
                    
                    # if image17_HTLF_OD is active this frame...
                    if image17_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTLF_OD is stopping this frame...
                    if image17_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image17_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTLF_OD.stopped')
                            # update status
                            image17_HTLF_OD.status = FINISHED
                            image17_HTLF_OD.setAutoDraw(False)
                    
                    # *image18_HTLF_OD* updates
                    
                    # if image18_HTLF_OD is starting this frame...
                    if image18_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTLF_OD.frameNStart = frameN  # exact frame index
                        image18_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image18_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTLF_OD.started')
                        # update status
                        image18_HTLF_OD.status = STARTED
                        image18_HTLF_OD.setAutoDraw(True)
                    
                    # if image18_HTLF_OD is active this frame...
                    if image18_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTLF_OD is stopping this frame...
                    if image18_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image18_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTLF_OD.stopped')
                            # update status
                            image18_HTLF_OD.status = FINISHED
                            image18_HTLF_OD.setAutoDraw(False)
                    
                    # *image19_HTLF_OD* updates
                    
                    # if image19_HTLF_OD is starting this frame...
                    if image19_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTLF_OD.frameNStart = frameN  # exact frame index
                        image19_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image19_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTLF_OD.started')
                        # update status
                        image19_HTLF_OD.status = STARTED
                        image19_HTLF_OD.setAutoDraw(True)
                    
                    # if image19_HTLF_OD is active this frame...
                    if image19_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTLF_OD is stopping this frame...
                    if image19_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image19_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTLF_OD.stopped')
                            # update status
                            image19_HTLF_OD.status = FINISHED
                            image19_HTLF_OD.setAutoDraw(False)
                    
                    # *image20_HTLF_OD* updates
                    
                    # if image20_HTLF_OD is starting this frame...
                    if image20_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTLF_OD.frameNStart = frameN  # exact frame index
                        image20_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image20_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTLF_OD.started')
                        # update status
                        image20_HTLF_OD.status = STARTED
                        image20_HTLF_OD.setAutoDraw(True)
                    
                    # if image20_HTLF_OD is active this frame...
                    if image20_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTLF_OD is stopping this frame...
                    if image20_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image20_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTLF_OD.stopped')
                            # update status
                            image20_HTLF_OD.status = FINISHED
                            image20_HTLF_OD.setAutoDraw(False)
                    
                    # *image21_HTLF_OD* updates
                    
                    # if image21_HTLF_OD is starting this frame...
                    if image21_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTLF_OD.frameNStart = frameN  # exact frame index
                        image21_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image21_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTLF_OD.started')
                        # update status
                        image21_HTLF_OD.status = STARTED
                        image21_HTLF_OD.setAutoDraw(True)
                    
                    # if image21_HTLF_OD is active this frame...
                    if image21_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTLF_OD is stopping this frame...
                    if image21_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image21_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTLF_OD.stopped')
                            # update status
                            image21_HTLF_OD.status = FINISHED
                            image21_HTLF_OD.setAutoDraw(False)
                    
                    # *image22_HTLF_OD* updates
                    
                    # if image22_HTLF_OD is starting this frame...
                    if image22_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTLF_OD.frameNStart = frameN  # exact frame index
                        image22_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image22_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTLF_OD.started')
                        # update status
                        image22_HTLF_OD.status = STARTED
                        image22_HTLF_OD.setAutoDraw(True)
                    
                    # if image22_HTLF_OD is active this frame...
                    if image22_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTLF_OD is stopping this frame...
                    if image22_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image22_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTLF_OD.stopped')
                            # update status
                            image22_HTLF_OD.status = FINISHED
                            image22_HTLF_OD.setAutoDraw(False)
                    
                    # *image23_HTLF_OD* updates
                    
                    # if image23_HTLF_OD is starting this frame...
                    if image23_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTLF_OD.frameNStart = frameN  # exact frame index
                        image23_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image23_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTLF_OD.started')
                        # update status
                        image23_HTLF_OD.status = STARTED
                        image23_HTLF_OD.setAutoDraw(True)
                    
                    # if image23_HTLF_OD is active this frame...
                    if image23_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTLF_OD is stopping this frame...
                    if image23_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTLF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image23_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTLF_OD.stopped')
                            # update status
                            image23_HTLF_OD.status = FINISHED
                            image23_HTLF_OD.setAutoDraw(False)
                    
                    # *image24_HTLF_OD* updates
                    
                    # if image24_HTLF_OD is starting this frame...
                    if image24_HTLF_OD.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTLF_OD.frameNStart = frameN  # exact frame index
                        image24_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image24_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTLF_OD.started')
                        # update status
                        image24_HTLF_OD.status = STARTED
                        image24_HTLF_OD.setAutoDraw(True)
                    
                    # if image24_HTLF_OD is active this frame...
                    if image24_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTLF_OD is stopping this frame...
                    if image24_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTLF_OD.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image24_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTLF_OD.stopped')
                            # update status
                            image24_HTLF_OD.status = FINISHED
                            image24_HTLF_OD.setAutoDraw(False)
                    
                    # *image_Frame_HTLF_OD* updates
                    
                    # if image_Frame_HTLF_OD is starting this frame...
                    if image_Frame_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTLF_OD.frameNStart = frameN  # exact frame index
                        image_Frame_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTLF_OD.started')
                        # update status
                        image_Frame_HTLF_OD.status = STARTED
                        image_Frame_HTLF_OD.setAutoDraw(True)
                    
                    # if image_Frame_HTLF_OD is active this frame...
                    if image_Frame_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTLF_OD is stopping this frame...
                    if image_Frame_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTLF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image_Frame_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTLF_OD.stopped')
                            # update status
                            image_Frame_HTLF_OD.status = FINISHED
                            image_Frame_HTLF_OD.setAutoDraw(False)
                    
                    # *image_Cue_HTLF_OD* updates
                    
                    # if image_Cue_HTLF_OD is starting this frame...
                    if image_Cue_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTLF_OD.frameNStart = frameN  # exact frame index
                        image_Cue_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTLF_OD.started')
                        # update status
                        image_Cue_HTLF_OD.status = STARTED
                        image_Cue_HTLF_OD.setAutoDraw(True)
                    
                    # if image_Cue_HTLF_OD is active this frame...
                    if image_Cue_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTLF_OD is stopping this frame...
                    if image_Cue_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTLF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image_Cue_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTLF_OD.stopped')
                            # update status
                            image_Cue_HTLF_OD.status = FINISHED
                            image_Cue_HTLF_OD.setAutoDraw(False)
                    
                    # *image_Cross_HTLF_OD* updates
                    
                    # if image_Cross_HTLF_OD is starting this frame...
                    if image_Cross_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTLF_OD.frameNStart = frameN  # exact frame index
                        image_Cross_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTLF_OD.started')
                        # update status
                        image_Cross_HTLF_OD.status = STARTED
                        image_Cross_HTLF_OD.setAutoDraw(True)
                    
                    # if image_Cross_HTLF_OD is active this frame...
                    if image_Cross_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTLF_OD is stopping this frame...
                    if image_Cross_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTLF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image_Cross_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTLF_OD.stopped')
                            # update status
                            image_Cross_HTLF_OD.status = FINISHED
                            image_Cross_HTLF_OD.setAutoDraw(False)
                    
                    # *image_Tar_HTLF_OD* updates
                    
                    # if image_Tar_HTLF_OD is starting this frame...
                    if image_Tar_HTLF_OD.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTLF_OD.frameNStart = frameN  # exact frame index
                        image_Tar_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTLF_OD.started')
                        # update status
                        image_Tar_HTLF_OD.status = STARTED
                        image_Tar_HTLF_OD.setAutoDraw(True)
                    
                    # if image_Tar_HTLF_OD is active this frame...
                    if image_Tar_HTLF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTLF_OD is stopping this frame...
                    if image_Tar_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTLF_OD.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END
                            # keep track of stop time/frame for later
                            image_Tar_HTLF_OD.tStop = t  # not accounting for scr refresh
                            image_Tar_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTLF_OD.stopped')
                            # update status
                            image_Tar_HTLF_OD.status = FINISHED
                            image_Tar_HTLF_OD.setAutoDraw(False)
                    
                    # *key_HTLF_OD* updates
                    waitOnFlip = False
                    
                    # if key_HTLF_OD is starting this frame...
                    if key_HTLF_OD.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTLF_OD.frameNStart = frameN  # exact frame index
                        key_HTLF_OD.tStart = t  # local t and not account for scr refresh
                        key_HTLF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTLF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTLF_OD.started')
                        # update status
                        key_HTLF_OD.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTLF_OD.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTLF_OD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTLF_OD is stopping this frame...
                    if key_HTLF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTLF_OD.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTLF_OD.tStop = t  # not accounting for scr refresh
                            key_HTLF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTLF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTLF_OD.stopped')
                            # update status
                            key_HTLF_OD.status = FINISHED
                            key_HTLF_OD.status = FINISHED
                    if key_HTLF_OD.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTLF_OD.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTLF_OD_allKeys.extend(theseKeys)
                        if len(_key_HTLF_OD_allKeys):
                            key_HTLF_OD.keys = _key_HTLF_OD_allKeys[0].name  # just the first key pressed
                            key_HTLF_OD.rt = _key_HTLF_OD_allKeys[0].rt
                            key_HTLF_OD.duration = _key_HTLF_OD_allKeys[0].duration
                            # was this correct?
                            ### CUSTOM START ###
                            #if (key_HTLF_OD.keys == str(corr_resp)) or (key_HTLF_OD.keys == corr_resp):
                            if (key_HTLF_OD.keys == str(my_corrResp_L_OD[my_outerloopcounter][my_TDA_L_OD_counter])) or (key_HTLF_OD.keys == my_corrResp_L_OD[my_outerloopcounter][my_TDA_L_OD_counter]): ## CUSTOM: Replaced corr_resp with own variable my_corrResp_*
                            
                                key_HTLF_OD.corr = 1
                            else:
                                key_HTLF_OD.corr = 0
                            ### CUSTOM END ###
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTLF_OD.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTLF_OD.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END
                
                # --- Ending Routine "HTLF_OD" ---
                for thisComponent in HTLF_OD.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTLF_OD
                HTLF_OD.tStop = globalClock.getTime(format='float')
                HTLF_OD.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTLF_OD.stopped', HTLF_OD.tStop)
                # check responses
                if key_HTLF_OD.keys in ['', [], None]:  # No response was made
                    key_HTLF_OD.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_L_OD[my_outerloopcounter][my_TDA_L_OD_counter]).lower() == 'none': ## CUSTOM START & END
                       key_HTLF_OD.corr = 1;  # correct non-response
                    else:
                       key_HTLF_OD.corr = 0;  # failed to respond (incorrectly)
                # store data for TRials_HTLF_OD (TrialHandler)
                TRials_HTLF_OD.addData('key_HTLF_OD.keys',key_HTLF_OD.keys)
                TRials_HTLF_OD.addData('key_HTLF_OD.corr', key_HTLF_OD.corr)
                if key_HTLF_OD.keys != None:  # we had a response
                    TRials_HTLF_OD.addData('key_HTLF_OD.rt', key_HTLF_OD.rt)
                    TRials_HTLF_OD.addData('key_HTLF_OD.duration', key_HTLF_OD.duration)
                try:
                    if image1_HTLF_OD.tStopRefresh is not None:
                        duration_val = image1_HTLF_OD.tStopRefresh - image1_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image1_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTLF_OD).__name__,
                        trial_type='image1_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image1_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTLF_OD.tStopRefresh is not None:
                        duration_val = image2_HTLF_OD.tStopRefresh - image2_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image2_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTLF_OD).__name__,
                        trial_type='image2_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image2_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTLF_OD.tStopRefresh is not None:
                        duration_val = image3_HTLF_OD.tStopRefresh - image3_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image3_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTLF_OD).__name__,
                        trial_type='image3_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image3_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTLF_OD.tStopRefresh is not None:
                        duration_val = image4_HTLF_OD.tStopRefresh - image4_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image4_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTLF_OD).__name__,
                        trial_type='image4_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image4_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTLF_OD.tStopRefresh is not None:
                        duration_val = image5_HTLF_OD.tStopRefresh - image5_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image5_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTLF_OD).__name__,
                        trial_type='image5_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image5_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTLF_OD.tStopRefresh is not None:
                        duration_val = image6_HTLF_OD.tStopRefresh - image6_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image6_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTLF_OD).__name__,
                        trial_type='image6_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image6_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTLF_OD.tStopRefresh is not None:
                        duration_val = image7_HTLF_OD.tStopRefresh - image7_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image7_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTLF_OD).__name__,
                        trial_type='image7_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image7_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTLF_OD.tStopRefresh is not None:
                        duration_val = image8_HTLF_OD.tStopRefresh - image8_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image8_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTLF_OD).__name__,
                        trial_type='image8_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image8_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTLF_OD.tStopRefresh is not None:
                        duration_val = image9_HTLF_OD.tStopRefresh - image9_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image9_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTLF_OD).__name__,
                        trial_type='image9_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image9_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTLF_OD.tStopRefresh is not None:
                        duration_val = image10_HTLF_OD.tStopRefresh - image10_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image10_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTLF_OD).__name__,
                        trial_type='image10_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image10_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTLF_OD.tStopRefresh is not None:
                        duration_val = image11_HTLF_OD.tStopRefresh - image11_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image11_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTLF_OD).__name__,
                        trial_type='image11_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image11_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTLF_OD.tStopRefresh is not None:
                        duration_val = image12_HTLF_OD.tStopRefresh - image12_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image12_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTLF_OD).__name__,
                        trial_type='image12_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image12_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTLF_OD.tStopRefresh is not None:
                        duration_val = image13_HTLF_OD.tStopRefresh - image13_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image13_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTLF_OD).__name__,
                        trial_type='image13_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image13_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTLF_OD.tStopRefresh is not None:
                        duration_val = image14_HTLF_OD.tStopRefresh - image14_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image14_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTLF_OD).__name__,
                        trial_type='image14_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image14_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTLF_OD.tStopRefresh is not None:
                        duration_val = image15_HTLF_OD.tStopRefresh - image15_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image15_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTLF_OD).__name__,
                        trial_type='image15_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image15_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTLF_OD.tStopRefresh is not None:
                        duration_val = image16_HTLF_OD.tStopRefresh - image16_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image16_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTLF_OD).__name__,
                        trial_type='image16_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image16_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTLF_OD.tStopRefresh is not None:
                        duration_val = image17_HTLF_OD.tStopRefresh - image17_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image17_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTLF_OD).__name__,
                        trial_type='image17_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image17_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTLF_OD.tStopRefresh is not None:
                        duration_val = image18_HTLF_OD.tStopRefresh - image18_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image18_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTLF_OD).__name__,
                        trial_type='image18_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image18_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTLF_OD.tStopRefresh is not None:
                        duration_val = image19_HTLF_OD.tStopRefresh - image19_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image19_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTLF_OD).__name__,
                        trial_type='image19_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image19_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTLF_OD.tStopRefresh is not None:
                        duration_val = image20_HTLF_OD.tStopRefresh - image20_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image20_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTLF_OD).__name__,
                        trial_type='image20_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image20_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTLF_OD.tStopRefresh is not None:
                        duration_val = image21_HTLF_OD.tStopRefresh - image21_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image21_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTLF_OD).__name__,
                        trial_type='image21_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image21_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTLF_OD.tStopRefresh is not None:
                        duration_val = image22_HTLF_OD.tStopRefresh - image22_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image22_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTLF_OD).__name__,
                        trial_type='image22_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image22_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTLF_OD.tStopRefresh is not None:
                        duration_val = image23_HTLF_OD.tStopRefresh - image23_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image23_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTLF_OD).__name__,
                        trial_type='image23_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image23_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTLF_OD.tStopRefresh is not None:
                        duration_val = image24_HTLF_OD.tStopRefresh - image24_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image24_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTLF_OD).__name__,
                        trial_type='image24_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image24_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTLF_OD.tStopRefresh is not None:
                        duration_val = image_Cue_HTLF_OD.tStopRefresh - image_Cue_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image_Cue_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTLF_OD).__name__,
                        trial_type='image_Cue_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image_Cue_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTLF_OD.tStopRefresh is not None:
                        duration_val = image_Tar_HTLF_OD.tStopRefresh - image_Tar_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - image_Tar_HTLF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTLF_OD).__name__,
                        trial_type='image_Tar_HTLF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_image_Tar_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTLF_OD.tStopRefresh is not None:
                        duration_val = key_HTLF_OD.tStopRefresh - key_HTLF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTLF_OD.stopped'] - key_HTLF_OD.tStartRefresh
                    if hasattr(key_HTLF_OD, 'rt'):
                        rt_val = key_HTLF_OD.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTLF_OD" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTLF_OD.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTLF_OD).__name__,
                        trial_type='key_HTLF_OD',
                        value=key_HTLF_OD.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        TRials_HTLF_OD.addData('bidsEvent_key_HTLF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTLF_OD.maxDurationReached:
                    routineTimer.addTime(-HTLF_OD.maxDuration)
                elif HTLF_OD.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                #--- CUSTOM START ---
                my_TDA_L_OD_counter = my_TDA_L_OD_counter + 1
                #--- CUSTOM END ---
                
            # completed 1.0 repeats of 'TRials_HTLF_OD'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if TRials_HTLF_OD.trialList in ([], [None], None):
                params = []
            else:
                params = TRials_HTLF_OD.trialList[0].keys()
            # save data for this loop
            TRials_HTLF_OD.saveAsExcel(filename + '.xlsx', sheetName='TRials_HTLF_OD',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            TRials_HTLF_OD.saveAsText(filename + 'TRials_HTLF_OD.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
        #Custom-Comment: End TDA_HTLF_OD
        elif my_seq_num[my_outerloopcounter]==6:
        #Custom-Comment: Start TDA_HTRF_OD
        
            # set up handler to look after randomisation of conditions etc
            Trials_HTRF_OD = data.TrialHandler2(
                name='Trials_HTRF_OD',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Conditions_TR_ND.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(Trials_HTRF_OD)  # add the loop to the experiment
            thisTrials_HTRF_OD = Trials_HTRF_OD.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRF_OD.rgb)
            if thisTrials_HTRF_OD != None:
                for paramName in thisTrials_HTRF_OD:
                    globals()[paramName] = thisTrials_HTRF_OD[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            #--- CUSTOM START ---
            my_TDA_R_OD_counter = 0
            #--- CUSTOM END ---
            
            for thisTrials_HTRF_OD in Trials_HTRF_OD:
                currentLoop = Trials_HTRF_OD
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrials_HTRF_OD.rgb)
                if thisTrials_HTRF_OD != None:
                    for paramName in thisTrials_HTRF_OD:
                        globals()[paramName] = thisTrials_HTRF_OD[paramName]
                
                # --- Prepare to start Routine "HTRF_OD" ---
                # create an object to store info about Routine HTRF_OD
                HTRF_OD = data.Routine(
                    name='HTRF_OD',
                    components=[image1_HTRF_OD, image2_HTRF_OD, image3_HTRF_OD, image4_HTRF_OD, image5_HTRF_OD, image6_HTRF_OD, image7_HTRF_OD, image8_HTRF_OD, image9_HTRF_OD, image10_HTRF_OD, image11_HTRF_OD, image12_HTRF_OD, image13_HTRF_OD, image14_HTRF_OD, image15_HTRF_OD, image16_HTRF_OD, image17_HTRF_OD, image18_HTRF_OD, image19_HTRF_OD, image20_HTRF_OD, image21_HTRF_OD, image22_HTRF_OD, image23_HTRF_OD, image24_HTRF_OD, image_Frame_HTRF_OD, image_Cue_HTRF_OD, image_Cross_HTRF_OD, image_Tar_HTRF_OD, key_HTRF_OD],
                )
                HTRF_OD.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                #--- CUSTOM START ---
                image_Tar_HTRF_OD.setPos((my_coordTDA_R_OD[0, my_TDA_R_OD_counter, my_outerloopcounter], my_coordTDA_R_OD[1, my_TDA_R_OD_counter, my_outerloopcounter]))
                # replaces the old line 'image_Tar_HTRF_OD.setPos((target_xcoor, target_ycoor))'
                #--- CUSTOM END ---
                # create starting attributes for key_HTRF_OD
                key_HTRF_OD.keys = []
                key_HTRF_OD.rt = []
                _key_HTRF_OD_allKeys = []
                # store start times for HTRF_OD
                HTRF_OD.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                HTRF_OD.tStart = globalClock.getTime(format='float')
                HTRF_OD.status = STARTED
                thisExp.addData('HTRF_OD.started', HTRF_OD.tStart)
                HTRF_OD.maxDuration = None
                # keep track of which components have finished
                HTRF_ODComponents = HTRF_OD.components
                for thisComponent in HTRF_OD.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "HTRF_OD" ---
                # if trial has changed, end Routine now
                if isinstance(Trials_HTRF_OD, data.TrialHandler2) and thisTrials_HTRF_OD.thisN != Trials_HTRF_OD.thisTrial.thisN:
                    continueRoutine = False
                HTRF_OD.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 1.6:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image1_HTRF_OD* updates
                    
                    # if image1_HTRF_OD is starting this frame...
                    if image1_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image1_HTRF_OD.frameNStart = frameN  # exact frame index
                        image1_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image1_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image1_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image1_HTRF_OD.started')
                        # update status
                        image1_HTRF_OD.status = STARTED
                        image1_HTRF_OD.setAutoDraw(True)
                    
                    # if image1_HTRF_OD is active this frame...
                    if image1_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image1_HTRF_OD is stopping this frame...
                    if image1_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image1_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image1_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image1_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image1_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image1_HTRF_OD.stopped')
                            # update status
                            image1_HTRF_OD.status = FINISHED
                            image1_HTRF_OD.setAutoDraw(False)
                    
                    # *image2_HTRF_OD* updates
                    
                    # if image2_HTRF_OD is starting this frame...
                    if image2_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.066667-frameTolerance:
                        # keep track of start time/frame for later
                        image2_HTRF_OD.frameNStart = frameN  # exact frame index
                        image2_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image2_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image2_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image2_HTRF_OD.started')
                        # update status
                        image2_HTRF_OD.status = STARTED
                        image2_HTRF_OD.setAutoDraw(True)
                    
                    # if image2_HTRF_OD is active this frame...
                    if image2_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image2_HTRF_OD is stopping this frame...
                    if image2_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image2_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image2_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image2_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image2_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image2_HTRF_OD.stopped')
                            # update status
                            image2_HTRF_OD.status = FINISHED
                            image2_HTRF_OD.setAutoDraw(False)
                    
                    # *image3_HTRF_OD* updates
                    
                    # if image3_HTRF_OD is starting this frame...
                    if image3_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.133334-frameTolerance:
                        # keep track of start time/frame for later
                        image3_HTRF_OD.frameNStart = frameN  # exact frame index
                        image3_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image3_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image3_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image3_HTRF_OD.started')
                        # update status
                        image3_HTRF_OD.status = STARTED
                        image3_HTRF_OD.setAutoDraw(True)
                    
                    # if image3_HTRF_OD is active this frame...
                    if image3_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image3_HTRF_OD is stopping this frame...
                    if image3_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image3_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image3_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image3_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image3_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image3_HTRF_OD.stopped')
                            # update status
                            image3_HTRF_OD.status = FINISHED
                            image3_HTRF_OD.setAutoDraw(False)
                    
                    # *image4_HTRF_OD* updates
                    
                    # if image4_HTRF_OD is starting this frame...
                    if image4_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.200001-frameTolerance:
                        # keep track of start time/frame for later
                        image4_HTRF_OD.frameNStart = frameN  # exact frame index
                        image4_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image4_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image4_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image4_HTRF_OD.started')
                        # update status
                        image4_HTRF_OD.status = STARTED
                        image4_HTRF_OD.setAutoDraw(True)
                    
                    # if image4_HTRF_OD is active this frame...
                    if image4_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image4_HTRF_OD is stopping this frame...
                    if image4_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image4_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image4_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image4_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image4_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image4_HTRF_OD.stopped')
                            # update status
                            image4_HTRF_OD.status = FINISHED
                            image4_HTRF_OD.setAutoDraw(False)
                    
                    # *image5_HTRF_OD* updates
                    
                    # if image5_HTRF_OD is starting this frame...
                    if image5_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.266668-frameTolerance:
                        # keep track of start time/frame for later
                        image5_HTRF_OD.frameNStart = frameN  # exact frame index
                        image5_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image5_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image5_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image5_HTRF_OD.started')
                        # update status
                        image5_HTRF_OD.status = STARTED
                        image5_HTRF_OD.setAutoDraw(True)
                    
                    # if image5_HTRF_OD is active this frame...
                    if image5_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image5_HTRF_OD is stopping this frame...
                    if image5_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image5_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image5_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image5_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image5_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image5_HTRF_OD.stopped')
                            # update status
                            image5_HTRF_OD.status = FINISHED
                            image5_HTRF_OD.setAutoDraw(False)
                    
                    # *image6_HTRF_OD* updates
                    
                    # if image6_HTRF_OD is starting this frame...
                    if image6_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.333335-frameTolerance:
                        # keep track of start time/frame for later
                        image6_HTRF_OD.frameNStart = frameN  # exact frame index
                        image6_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image6_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image6_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image6_HTRF_OD.started')
                        # update status
                        image6_HTRF_OD.status = STARTED
                        image6_HTRF_OD.setAutoDraw(True)
                    
                    # if image6_HTRF_OD is active this frame...
                    if image6_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image6_HTRF_OD is stopping this frame...
                    if image6_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image6_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image6_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image6_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image6_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image6_HTRF_OD.stopped')
                            # update status
                            image6_HTRF_OD.status = FINISHED
                            image6_HTRF_OD.setAutoDraw(False)
                    
                    # *image7_HTRF_OD* updates
                    
                    # if image7_HTRF_OD is starting this frame...
                    if image7_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.400002-frameTolerance:
                        # keep track of start time/frame for later
                        image7_HTRF_OD.frameNStart = frameN  # exact frame index
                        image7_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image7_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image7_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image7_HTRF_OD.started')
                        # update status
                        image7_HTRF_OD.status = STARTED
                        image7_HTRF_OD.setAutoDraw(True)
                    
                    # if image7_HTRF_OD is active this frame...
                    if image7_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image7_HTRF_OD is stopping this frame...
                    if image7_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image7_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image7_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image7_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image7_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image7_HTRF_OD.stopped')
                            # update status
                            image7_HTRF_OD.status = FINISHED
                            image7_HTRF_OD.setAutoDraw(False)
                    
                    # *image8_HTRF_OD* updates
                    
                    # if image8_HTRF_OD is starting this frame...
                    if image8_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.466669-frameTolerance:
                        # keep track of start time/frame for later
                        image8_HTRF_OD.frameNStart = frameN  # exact frame index
                        image8_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image8_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image8_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image8_HTRF_OD.started')
                        # update status
                        image8_HTRF_OD.status = STARTED
                        image8_HTRF_OD.setAutoDraw(True)
                    
                    # if image8_HTRF_OD is active this frame...
                    if image8_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image8_HTRF_OD is stopping this frame...
                    if image8_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image8_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image8_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image8_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image8_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image8_HTRF_OD.stopped')
                            # update status
                            image8_HTRF_OD.status = FINISHED
                            image8_HTRF_OD.setAutoDraw(False)
                    
                    # *image9_HTRF_OD* updates
                    
                    # if image9_HTRF_OD is starting this frame...
                    if image9_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.533336-frameTolerance:
                        # keep track of start time/frame for later
                        image9_HTRF_OD.frameNStart = frameN  # exact frame index
                        image9_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image9_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image9_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image9_HTRF_OD.started')
                        # update status
                        image9_HTRF_OD.status = STARTED
                        image9_HTRF_OD.setAutoDraw(True)
                    
                    # if image9_HTRF_OD is active this frame...
                    if image9_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image9_HTRF_OD is stopping this frame...
                    if image9_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image9_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image9_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image9_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image9_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image9_HTRF_OD.stopped')
                            # update status
                            image9_HTRF_OD.status = FINISHED
                            image9_HTRF_OD.setAutoDraw(False)
                    
                    # *image10_HTRF_OD* updates
                    
                    # if image10_HTRF_OD is starting this frame...
                    if image10_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.600003-frameTolerance:
                        # keep track of start time/frame for later
                        image10_HTRF_OD.frameNStart = frameN  # exact frame index
                        image10_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image10_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image10_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image10_HTRF_OD.started')
                        # update status
                        image10_HTRF_OD.status = STARTED
                        image10_HTRF_OD.setAutoDraw(True)
                    
                    # if image10_HTRF_OD is active this frame...
                    if image10_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image10_HTRF_OD is stopping this frame...
                    if image10_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image10_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image10_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image10_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image10_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image10_HTRF_OD.stopped')
                            # update status
                            image10_HTRF_OD.status = FINISHED
                            image10_HTRF_OD.setAutoDraw(False)
                    
                    # *image11_HTRF_OD* updates
                    
                    # if image11_HTRF_OD is starting this frame...
                    if image11_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.666670-frameTolerance:
                        # keep track of start time/frame for later
                        image11_HTRF_OD.frameNStart = frameN  # exact frame index
                        image11_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image11_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image11_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image11_HTRF_OD.started')
                        # update status
                        image11_HTRF_OD.status = STARTED
                        image11_HTRF_OD.setAutoDraw(True)
                    
                    # if image11_HTRF_OD is active this frame...
                    if image11_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image11_HTRF_OD is stopping this frame...
                    if image11_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image11_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image11_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image11_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image11_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image11_HTRF_OD.stopped')
                            # update status
                            image11_HTRF_OD.status = FINISHED
                            image11_HTRF_OD.setAutoDraw(False)
                    
                    # *image12_HTRF_OD* updates
                    
                    # if image12_HTRF_OD is starting this frame...
                    if image12_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.733337-frameTolerance:
                        # keep track of start time/frame for later
                        image12_HTRF_OD.frameNStart = frameN  # exact frame index
                        image12_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image12_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image12_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image12_HTRF_OD.started')
                        # update status
                        image12_HTRF_OD.status = STARTED
                        image12_HTRF_OD.setAutoDraw(True)
                    
                    # if image12_HTRF_OD is active this frame...
                    if image12_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image12_HTRF_OD is stopping this frame...
                    if image12_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image12_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image12_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image12_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image12_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image12_HTRF_OD.stopped')
                            # update status
                            image12_HTRF_OD.status = FINISHED
                            image12_HTRF_OD.setAutoDraw(False)
                    
                    # *image13_HTRF_OD* updates
                    
                    # if image13_HTRF_OD is starting this frame...
                    if image13_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.800004-frameTolerance:
                        # keep track of start time/frame for later
                        image13_HTRF_OD.frameNStart = frameN  # exact frame index
                        image13_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image13_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image13_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image13_HTRF_OD.started')
                        # update status
                        image13_HTRF_OD.status = STARTED
                        image13_HTRF_OD.setAutoDraw(True)
                    
                    # if image13_HTRF_OD is active this frame...
                    if image13_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image13_HTRF_OD is stopping this frame...
                    if image13_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image13_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image13_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image13_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image13_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image13_HTRF_OD.stopped')
                            # update status
                            image13_HTRF_OD.status = FINISHED
                            image13_HTRF_OD.setAutoDraw(False)
                    
                    # *image14_HTRF_OD* updates
                    
                    # if image14_HTRF_OD is starting this frame...
                    if image14_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.866671-frameTolerance:
                        # keep track of start time/frame for later
                        image14_HTRF_OD.frameNStart = frameN  # exact frame index
                        image14_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image14_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image14_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image14_HTRF_OD.started')
                        # update status
                        image14_HTRF_OD.status = STARTED
                        image14_HTRF_OD.setAutoDraw(True)
                    
                    # if image14_HTRF_OD is active this frame...
                    if image14_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image14_HTRF_OD is stopping this frame...
                    if image14_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image14_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image14_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image14_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image14_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image14_HTRF_OD.stopped')
                            # update status
                            image14_HTRF_OD.status = FINISHED
                            image14_HTRF_OD.setAutoDraw(False)
                    
                    # *image15_HTRF_OD* updates
                    
                    # if image15_HTRF_OD is starting this frame...
                    if image15_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.933338-frameTolerance:
                        # keep track of start time/frame for later
                        image15_HTRF_OD.frameNStart = frameN  # exact frame index
                        image15_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image15_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image15_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image15_HTRF_OD.started')
                        # update status
                        image15_HTRF_OD.status = STARTED
                        image15_HTRF_OD.setAutoDraw(True)
                    
                    # if image15_HTRF_OD is active this frame...
                    if image15_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image15_HTRF_OD is stopping this frame...
                    if image15_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image15_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image15_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image15_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image15_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image15_HTRF_OD.stopped')
                            # update status
                            image15_HTRF_OD.status = FINISHED
                            image15_HTRF_OD.setAutoDraw(False)
                    
                    # *image16_HTRF_OD* updates
                    
                    # if image16_HTRF_OD is starting this frame...
                    if image16_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.000005-frameTolerance:
                        # keep track of start time/frame for later
                        image16_HTRF_OD.frameNStart = frameN  # exact frame index
                        image16_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image16_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image16_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image16_HTRF_OD.started')
                        # update status
                        image16_HTRF_OD.status = STARTED
                        image16_HTRF_OD.setAutoDraw(True)
                    
                    # if image16_HTRF_OD is active this frame...
                    if image16_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image16_HTRF_OD is stopping this frame...
                    if image16_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image16_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image16_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image16_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image16_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image16_HTRF_OD.stopped')
                            # update status
                            image16_HTRF_OD.status = FINISHED
                            image16_HTRF_OD.setAutoDraw(False)
                    
                    # *image17_HTRF_OD* updates
                    
                    # if image17_HTRF_OD is starting this frame...
                    if image17_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.066672-frameTolerance:
                        # keep track of start time/frame for later
                        image17_HTRF_OD.frameNStart = frameN  # exact frame index
                        image17_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image17_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image17_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image17_HTRF_OD.started')
                        # update status
                        image17_HTRF_OD.status = STARTED
                        image17_HTRF_OD.setAutoDraw(True)
                    
                    # if image17_HTRF_OD is active this frame...
                    if image17_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image17_HTRF_OD is stopping this frame...
                    if image17_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image17_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image17_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image17_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image17_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image17_HTRF_OD.stopped')
                            # update status
                            image17_HTRF_OD.status = FINISHED
                            image17_HTRF_OD.setAutoDraw(False)
                    
                    # *image18_HTRF_OD* updates
                    
                    # if image18_HTRF_OD is starting this frame...
                    if image18_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.133339-frameTolerance:
                        # keep track of start time/frame for later
                        image18_HTRF_OD.frameNStart = frameN  # exact frame index
                        image18_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image18_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image18_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image18_HTRF_OD.started')
                        # update status
                        image18_HTRF_OD.status = STARTED
                        image18_HTRF_OD.setAutoDraw(True)
                    
                    # if image18_HTRF_OD is active this frame...
                    if image18_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image18_HTRF_OD is stopping this frame...
                    if image18_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image18_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image18_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image18_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image18_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image18_HTRF_OD.stopped')
                            # update status
                            image18_HTRF_OD.status = FINISHED
                            image18_HTRF_OD.setAutoDraw(False)
                    
                    # *image19_HTRF_OD* updates
                    
                    # if image19_HTRF_OD is starting this frame...
                    if image19_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.200006-frameTolerance:
                        # keep track of start time/frame for later
                        image19_HTRF_OD.frameNStart = frameN  # exact frame index
                        image19_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image19_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image19_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image19_HTRF_OD.started')
                        # update status
                        image19_HTRF_OD.status = STARTED
                        image19_HTRF_OD.setAutoDraw(True)
                    
                    # if image19_HTRF_OD is active this frame...
                    if image19_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image19_HTRF_OD is stopping this frame...
                    if image19_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image19_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image19_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image19_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image19_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image19_HTRF_OD.stopped')
                            # update status
                            image19_HTRF_OD.status = FINISHED
                            image19_HTRF_OD.setAutoDraw(False)
                    
                    # *image20_HTRF_OD* updates
                    
                    # if image20_HTRF_OD is starting this frame...
                    if image20_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.266673-frameTolerance:
                        # keep track of start time/frame for later
                        image20_HTRF_OD.frameNStart = frameN  # exact frame index
                        image20_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image20_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image20_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image20_HTRF_OD.started')
                        # update status
                        image20_HTRF_OD.status = STARTED
                        image20_HTRF_OD.setAutoDraw(True)
                    
                    # if image20_HTRF_OD is active this frame...
                    if image20_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image20_HTRF_OD is stopping this frame...
                    if image20_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image20_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image20_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image20_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image20_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image20_HTRF_OD.stopped')
                            # update status
                            image20_HTRF_OD.status = FINISHED
                            image20_HTRF_OD.setAutoDraw(False)
                    
                    # *image21_HTRF_OD* updates
                    
                    # if image21_HTRF_OD is starting this frame...
                    if image21_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.333340-frameTolerance:
                        # keep track of start time/frame for later
                        image21_HTRF_OD.frameNStart = frameN  # exact frame index
                        image21_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image21_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image21_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image21_HTRF_OD.started')
                        # update status
                        image21_HTRF_OD.status = STARTED
                        image21_HTRF_OD.setAutoDraw(True)
                    
                    # if image21_HTRF_OD is active this frame...
                    if image21_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image21_HTRF_OD is stopping this frame...
                    if image21_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image21_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image21_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image21_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image21_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image21_HTRF_OD.stopped')
                            # update status
                            image21_HTRF_OD.status = FINISHED
                            image21_HTRF_OD.setAutoDraw(False)
                    
                    # *image22_HTRF_OD* updates
                    
                    # if image22_HTRF_OD is starting this frame...
                    if image22_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.400007-frameTolerance:
                        # keep track of start time/frame for later
                        image22_HTRF_OD.frameNStart = frameN  # exact frame index
                        image22_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image22_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image22_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image22_HTRF_OD.started')
                        # update status
                        image22_HTRF_OD.status = STARTED
                        image22_HTRF_OD.setAutoDraw(True)
                    
                    # if image22_HTRF_OD is active this frame...
                    if image22_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image22_HTRF_OD is stopping this frame...
                    if image22_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image22_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image22_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image22_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image22_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image22_HTRF_OD.stopped')
                            # update status
                            image22_HTRF_OD.status = FINISHED
                            image22_HTRF_OD.setAutoDraw(False)
                    
                    # *image23_HTRF_OD* updates
                    
                    # if image23_HTRF_OD is starting this frame...
                    if image23_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.466674-frameTolerance:
                        # keep track of start time/frame for later
                        image23_HTRF_OD.frameNStart = frameN  # exact frame index
                        image23_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image23_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image23_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image23_HTRF_OD.started')
                        # update status
                        image23_HTRF_OD.status = STARTED
                        image23_HTRF_OD.setAutoDraw(True)
                    
                    # if image23_HTRF_OD is active this frame...
                    if image23_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image23_HTRF_OD is stopping this frame...
                    if image23_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image23_HTRF_OD.tStartRefresh + 0.066667-frameTolerance:
                            # keep track of stop time/frame for later
                            image23_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image23_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image23_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image23_HTRF_OD.stopped')
                            # update status
                            image23_HTRF_OD.status = FINISHED
                            image23_HTRF_OD.setAutoDraw(False)
                    
                    # *image24_HTRF_OD* updates
                    
                    # if image24_HTRF_OD is starting this frame...
                    if image24_HTRF_OD.status == NOT_STARTED and tThisFlip >= 1.533341-frameTolerance:
                        # keep track of start time/frame for later
                        image24_HTRF_OD.frameNStart = frameN  # exact frame index
                        image24_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image24_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image24_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image24_HTRF_OD.started')
                        # update status
                        image24_HTRF_OD.status = STARTED
                        image24_HTRF_OD.setAutoDraw(True)
                    
                    # if image24_HTRF_OD is active this frame...
                    if image24_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image24_HTRF_OD is stopping this frame...
                    if image24_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image24_HTRF_OD.tStartRefresh + 0.066659-frameTolerance:
                            # keep track of stop time/frame for later
                            image24_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image24_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image24_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image24_HTRF_OD.stopped')
                            # update status
                            image24_HTRF_OD.status = FINISHED
                            image24_HTRF_OD.setAutoDraw(False)
                    
                    # *image_Frame_HTRF_OD* updates
                    
                    # if image_Frame_HTRF_OD is starting this frame...
                    if image_Frame_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Frame_HTRF_OD.frameNStart = frameN  # exact frame index
                        image_Frame_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image_Frame_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Frame_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_HTRF_OD.started')
                        # update status
                        image_Frame_HTRF_OD.status = STARTED
                        image_Frame_HTRF_OD.setAutoDraw(True)
                    
                    # if image_Frame_HTRF_OD is active this frame...
                    if image_Frame_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Frame_HTRF_OD is stopping this frame...
                    if image_Frame_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Frame_HTRF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Frame_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image_Frame_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Frame_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Frame_HTRF_OD.stopped')
                            # update status
                            image_Frame_HTRF_OD.status = FINISHED
                            image_Frame_HTRF_OD.setAutoDraw(False)
                    
                    # *image_Cue_HTRF_OD* updates
                    
                    # if image_Cue_HTRF_OD is starting this frame...
                    if image_Cue_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cue_HTRF_OD.frameNStart = frameN  # exact frame index
                        image_Cue_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image_Cue_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cue_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cue_HTRF_OD.started')
                        # update status
                        image_Cue_HTRF_OD.status = STARTED
                        image_Cue_HTRF_OD.setAutoDraw(True)
                    
                    # if image_Cue_HTRF_OD is active this frame...
                    if image_Cue_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cue_HTRF_OD is stopping this frame...
                    if image_Cue_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cue_HTRF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cue_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image_Cue_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cue_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cue_HTRF_OD.stopped')
                            # update status
                            image_Cue_HTRF_OD.status = FINISHED
                            image_Cue_HTRF_OD.setAutoDraw(False)
                    
                    # *image_Cross_HTRF_OD* updates
                    
                    # if image_Cross_HTRF_OD is starting this frame...
                    if image_Cross_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image_Cross_HTRF_OD.frameNStart = frameN  # exact frame index
                        image_Cross_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image_Cross_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Cross_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_HTRF_OD.started')
                        # update status
                        image_Cross_HTRF_OD.status = STARTED
                        image_Cross_HTRF_OD.setAutoDraw(True)
                    
                    # if image_Cross_HTRF_OD is active this frame...
                    if image_Cross_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Cross_HTRF_OD is stopping this frame...
                    if image_Cross_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Cross_HTRF_OD.tStartRefresh + 1.6-frameTolerance:
                            # keep track of stop time/frame for later
                            image_Cross_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image_Cross_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Cross_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Cross_HTRF_OD.stopped')
                            # update status
                            image_Cross_HTRF_OD.status = FINISHED
                            image_Cross_HTRF_OD.setAutoDraw(False)
                    
                    # *image_Tar_HTRF_OD* updates
                    
                    # if image_Tar_HTRF_OD is starting this frame...
                    if image_Tar_HTRF_OD.status == NOT_STARTED and tThisFlip >= (1.6 - my_randtartime[my_randtartime_counter])-frameTolerance:   ### CUSTOM START AND END 
                        # keep track of start time/frame for later
                        image_Tar_HTRF_OD.frameNStart = frameN  # exact frame index
                        image_Tar_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        image_Tar_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_Tar_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Tar_HTRF_OD.started')
                        # update status
                        image_Tar_HTRF_OD.status = STARTED
                        image_Tar_HTRF_OD.setAutoDraw(True)
                    
                    # if image_Tar_HTRF_OD is active this frame...
                    if image_Tar_HTRF_OD.status == STARTED:
                        # update params
                        pass
                    
                    # if image_Tar_HTRF_OD is stopping this frame...
                    if image_Tar_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_Tar_HTRF_OD.tStartRefresh + my_randtartime[my_randtartime_counter]-frameTolerance:         ### CUSTOM START AND END 
                            # keep track of stop time/frame for later
                            image_Tar_HTRF_OD.tStop = t  # not accounting for scr refresh
                            image_Tar_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            image_Tar_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image_Tar_HTRF_OD.stopped')
                            # update status
                            image_Tar_HTRF_OD.status = FINISHED
                            image_Tar_HTRF_OD.setAutoDraw(False)
                    
                    # *key_HTRF_OD* updates
                    waitOnFlip = False
                    
                    # if key_HTRF_OD is starting this frame...
                    if key_HTRF_OD.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        key_HTRF_OD.frameNStart = frameN  # exact frame index
                        key_HTRF_OD.tStart = t  # local t and not account for scr refresh
                        key_HTRF_OD.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_HTRF_OD, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_HTRF_OD.started')
                        # update status
                        key_HTRF_OD.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_HTRF_OD.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_HTRF_OD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_HTRF_OD is stopping this frame...
                    if key_HTRF_OD.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_HTRF_OD.tStartRefresh + 1.5-frameTolerance:
                            # keep track of stop time/frame for later
                            key_HTRF_OD.tStop = t  # not accounting for scr refresh
                            key_HTRF_OD.tStopRefresh = tThisFlipGlobal  # on global time
                            key_HTRF_OD.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_HTRF_OD.stopped')
                            # update status
                            key_HTRF_OD.status = FINISHED
                            key_HTRF_OD.status = FINISHED
                    if key_HTRF_OD.status == STARTED and not waitOnFlip:
                        theseKeys = key_HTRF_OD.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                        _key_HTRF_OD_allKeys.extend(theseKeys)
                        if len(_key_HTRF_OD_allKeys):
                            key_HTRF_OD.keys = _key_HTRF_OD_allKeys[0].name  # just the first key pressed
                            key_HTRF_OD.rt = _key_HTRF_OD_allKeys[0].rt
                            key_HTRF_OD.duration = _key_HTRF_OD_allKeys[0].duration
                            # was this correct?
                            ### CUSTOM START 
                            if (key_HTRF_OD.keys == str(my_corrResp_R_OD[my_outerloopcounter][my_TDA_R_OD_counter])) or (key_HTRF_OD.keys == my_corrResp_R_OD[my_outerloopcounter][my_TDA_R_OD_counter]):
                            #if (key_HTRF_OD.keys == str(corr_resp)) or (key_HTRF_OD.keys == corr_resp):
                                key_HTRF_OD.corr = 1
                            else:
                                key_HTRF_OD.corr = 0
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        HTRF_OD.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in HTRF_OD.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                my_randtartime_counter = my_randtartime_counter + 1          ### CUSTOM START AND END 
                
                # --- Ending Routine "HTRF_OD" ---
                for thisComponent in HTRF_OD.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for HTRF_OD
                HTRF_OD.tStop = globalClock.getTime(format='float')
                HTRF_OD.tStopRefresh = tThisFlipGlobal
                thisExp.addData('HTRF_OD.stopped', HTRF_OD.tStop)
                # check responses
                if key_HTRF_OD.keys in ['', [], None]:  # No response was made
                    key_HTRF_OD.keys = None
                    # was no response the correct answer?!
                    if str(my_corrResp_R_OD[my_outerloopcounter][my_TDA_R_OD_counter]).lower() == 'none': ## CUSTOM START & END
                       key_HTRF_OD.corr = 1;  # correct non-response
                    else:
                       key_HTRF_OD.corr = 0;  # failed to respond (incorrectly)
                # store data for Trials_HTRF_OD (TrialHandler)
                Trials_HTRF_OD.addData('key_HTRF_OD.keys',key_HTRF_OD.keys)
                Trials_HTRF_OD.addData('key_HTRF_OD.corr', key_HTRF_OD.corr)
                if key_HTRF_OD.keys != None:  # we had a response
                    Trials_HTRF_OD.addData('key_HTRF_OD.rt', key_HTRF_OD.rt)
                    Trials_HTRF_OD.addData('key_HTRF_OD.duration', key_HTRF_OD.duration)
                try:
                    if image1_HTRF_OD.tStopRefresh is not None:
                        duration_val = image1_HTRF_OD.tStopRefresh - image1_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image1_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image1_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image1_HTRF_OD).__name__,
                        trial_type='image1_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image1_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image2_HTRF_OD.tStopRefresh is not None:
                        duration_val = image2_HTRF_OD.tStopRefresh - image2_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image2_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image2_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image2_HTRF_OD).__name__,
                        trial_type='image2_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image2_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image3_HTRF_OD.tStopRefresh is not None:
                        duration_val = image3_HTRF_OD.tStopRefresh - image3_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image3_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image3_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image3_HTRF_OD).__name__,
                        trial_type='image3_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image3_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image4_HTRF_OD.tStopRefresh is not None:
                        duration_val = image4_HTRF_OD.tStopRefresh - image4_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image4_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image4_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image4_HTRF_OD).__name__,
                        trial_type='image4_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image4_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image5_HTRF_OD.tStopRefresh is not None:
                        duration_val = image5_HTRF_OD.tStopRefresh - image5_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image5_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image5_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image5_HTRF_OD).__name__,
                        trial_type='image5_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image5_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image6_HTRF_OD.tStopRefresh is not None:
                        duration_val = image6_HTRF_OD.tStopRefresh - image6_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image6_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image6_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image6_HTRF_OD).__name__,
                        trial_type='image6_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image6_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image7_HTRF_OD.tStopRefresh is not None:
                        duration_val = image7_HTRF_OD.tStopRefresh - image7_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image7_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image7_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image7_HTRF_OD).__name__,
                        trial_type='image7_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image7_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image8_HTRF_OD.tStopRefresh is not None:
                        duration_val = image8_HTRF_OD.tStopRefresh - image8_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image8_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image8_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image8_HTRF_OD).__name__,
                        trial_type='image8_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image8_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image9_HTRF_OD.tStopRefresh is not None:
                        duration_val = image9_HTRF_OD.tStopRefresh - image9_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image9_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image9_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image9_HTRF_OD).__name__,
                        trial_type='image9_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image9_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image10_HTRF_OD.tStopRefresh is not None:
                        duration_val = image10_HTRF_OD.tStopRefresh - image10_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image10_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image10_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image10_HTRF_OD).__name__,
                        trial_type='image10_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image10_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image11_HTRF_OD.tStopRefresh is not None:
                        duration_val = image11_HTRF_OD.tStopRefresh - image11_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image11_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image11_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image11_HTRF_OD).__name__,
                        trial_type='image11_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image11_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image12_HTRF_OD.tStopRefresh is not None:
                        duration_val = image12_HTRF_OD.tStopRefresh - image12_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image12_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image12_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image12_HTRF_OD).__name__,
                        trial_type='image12_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image12_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image13_HTRF_OD.tStopRefresh is not None:
                        duration_val = image13_HTRF_OD.tStopRefresh - image13_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image13_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image13_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image13_HTRF_OD).__name__,
                        trial_type='image13_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image13_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image14_HTRF_OD.tStopRefresh is not None:
                        duration_val = image14_HTRF_OD.tStopRefresh - image14_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image14_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image14_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image14_HTRF_OD).__name__,
                        trial_type='image14_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image14_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image15_HTRF_OD.tStopRefresh is not None:
                        duration_val = image15_HTRF_OD.tStopRefresh - image15_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image15_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image15_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image15_HTRF_OD).__name__,
                        trial_type='image15_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image15_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image16_HTRF_OD.tStopRefresh is not None:
                        duration_val = image16_HTRF_OD.tStopRefresh - image16_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image16_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image16_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image16_HTRF_OD).__name__,
                        trial_type='image16_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image16_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image17_HTRF_OD.tStopRefresh is not None:
                        duration_val = image17_HTRF_OD.tStopRefresh - image17_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image17_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image17_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image17_HTRF_OD).__name__,
                        trial_type='image17_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image17_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image18_HTRF_OD.tStopRefresh is not None:
                        duration_val = image18_HTRF_OD.tStopRefresh - image18_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image18_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image18_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image18_HTRF_OD).__name__,
                        trial_type='image18_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image18_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image19_HTRF_OD.tStopRefresh is not None:
                        duration_val = image19_HTRF_OD.tStopRefresh - image19_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image19_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image19_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image19_HTRF_OD).__name__,
                        trial_type='image19_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image19_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image20_HTRF_OD.tStopRefresh is not None:
                        duration_val = image20_HTRF_OD.tStopRefresh - image20_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image20_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image20_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image20_HTRF_OD).__name__,
                        trial_type='image20_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image20_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image21_HTRF_OD.tStopRefresh is not None:
                        duration_val = image21_HTRF_OD.tStopRefresh - image21_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image21_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image21_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image21_HTRF_OD).__name__,
                        trial_type='image21_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image21_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image22_HTRF_OD.tStopRefresh is not None:
                        duration_val = image22_HTRF_OD.tStopRefresh - image22_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image22_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image22_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image22_HTRF_OD).__name__,
                        trial_type='image22_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image22_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image23_HTRF_OD.tStopRefresh is not None:
                        duration_val = image23_HTRF_OD.tStopRefresh - image23_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image23_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image23_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image23_HTRF_OD).__name__,
                        trial_type='image23_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image23_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image24_HTRF_OD.tStopRefresh is not None:
                        duration_val = image24_HTRF_OD.tStopRefresh - image24_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image24_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image24_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image24_HTRF_OD).__name__,
                        trial_type='image24_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image24_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Cue_HTRF_OD.tStopRefresh is not None:
                        duration_val = image_Cue_HTRF_OD.tStopRefresh - image_Cue_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image_Cue_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Cue_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Cue_HTRF_OD).__name__,
                        trial_type='image_Cue_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image_Cue_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if image_Tar_HTRF_OD.tStopRefresh is not None:
                        duration_val = image_Tar_HTRF_OD.tStopRefresh - image_Tar_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - image_Tar_HTRF_OD.tStartRefresh
                    bids_event = BIDSTaskEvent(
                        onset=image_Tar_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        event_type=type(image_Tar_HTRF_OD).__name__,
                        trial_type='image_Tar_HTRF_OD',
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_image_Tar_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                try:
                    if key_HTRF_OD.tStopRefresh is not None:
                        duration_val = key_HTRF_OD.tStopRefresh - key_HTRF_OD.tStartRefresh
                    else:
                        duration_val = thisExp.thisEntry['HTRF_OD.stopped'] - key_HTRF_OD.tStartRefresh
                    if hasattr(key_HTRF_OD, 'rt'):
                        rt_val = key_HTRF_OD.rt
                    else:
                        rt_val = None
                        logging.warning('The linked component "key_HTRF_OD" does not have a reaction time(.rt) attribute. Unable to link BIDS response_time to this component. Please verify the component settings.')
                    bids_event = BIDSTaskEvent(
                        onset=key_HTRF_OD.tStartRefresh,
                        duration=duration_val,
                        response_time=rt_val,
                        event_type=type(key_HTRF_OD).__name__,
                        trial_type='key_HTRF_OD',
                        value=key_HTRF_OD.corr,
                    )
                    if bids_handler:
                        bids_handler.addEvent(bids_event)
                    else:
                        Trials_HTRF_OD.addData('bidsEvent_key_HTRF_OD.event', bids_event)
                except BIDSError as e:
                    print(f"[psychopy-bids(event)] An error occurred when creating BIDS event: {e}")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if HTRF_OD.maxDurationReached:
                    routineTimer.addTime(-HTRF_OD.maxDuration)
                elif HTRF_OD.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-1.600000)
                thisExp.nextEntry()
                
                ## CUSTOM START
                my_TDA_R_OD_counter = my_TDA_R_OD_counter + 1
                ## CUSTOM END
                
            # completed 1.0 repeats of 'Trials_HTRF_OD'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if Trials_HTRF_OD.trialList in ([], [None], None):
                params = []
            else:
                params = Trials_HTRF_OD.trialList[0].keys()
            # save data for this loop
            Trials_HTRF_OD.saveAsExcel(filename + '.xlsx', sheetName='Trials_HTRF_OD',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            Trials_HTRF_OD.saveAsText(filename + 'Trials_HTRF_OD.csv', delim=',',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            
            # --- Prepare to start Routine "Break" ---
            # create an object to store info about Routine Break
            Break = data.Routine(
                name='Break',
                components=[image_Frame_Break, image_Cross_Break],
            )
            Break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Break
            Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Break.tStart = globalClock.getTime(format='float')
            Break.status = STARTED
            thisExp.addData('Break.started', Break.tStart)
            Break.maxDuration = None
            # keep track of which components have finished
            BreakComponents = Break.components
            for thisComponent in Break.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Break" ---
            # if trial has changed, end Routine now
            if isinstance(Outerloop, data.TrialHandler2) and thisOuterloop.thisN != Outerloop.thisTrial.thisN:
                continueRoutine = False
            Break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_Frame_Break* updates
                
                # if image_Frame_Break is starting this frame...
                if image_Frame_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Frame_Break.frameNStart = frameN  # exact frame index
                    image_Frame_Break.tStart = t  # local t and not account for scr refresh
                    image_Frame_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Frame_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Frame_Break.started')
                    # update status
                    image_Frame_Break.status = STARTED
                    image_Frame_Break.setAutoDraw(True)
                
                # if image_Frame_Break is active this frame...
                if image_Frame_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Frame_Break is stopping this frame...
                if image_Frame_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Frame_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Frame_Break.tStop = t  # not accounting for scr refresh
                        image_Frame_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Frame_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Frame_Break.stopped')
                        # update status
                        image_Frame_Break.status = FINISHED
                        image_Frame_Break.setAutoDraw(False)
                
                # *image_Cross_Break* updates
                
                # if image_Cross_Break is starting this frame...
                if image_Cross_Break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_Cross_Break.frameNStart = frameN  # exact frame index
                    image_Cross_Break.tStart = t  # local t and not account for scr refresh
                    image_Cross_Break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Cross_Break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Cross_Break.started')
                    # update status
                    image_Cross_Break.status = STARTED
                    image_Cross_Break.setAutoDraw(True)
                
                # if image_Cross_Break is active this frame...
                if image_Cross_Break.status == STARTED:
                    # update params
                    pass
                
                # if image_Cross_Break is stopping this frame...
                if image_Cross_Break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_Cross_Break.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Cross_Break.tStop = t  # not accounting for scr refresh
                        image_Cross_Break.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Cross_Break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Cross_Break.stopped')
                        # update status
                        image_Cross_Break.status = FINISHED
                        image_Cross_Break.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Break" ---
            for thisComponent in Break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Break
            Break.tStop = globalClock.getTime(format='float')
            Break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Break.stopped', Break.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Break.maxDurationReached:
                routineTimer.addTime(-Break.maxDuration)
            elif Break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
                
        
        #Custom-Comment: End TDA_GR    
        
        #--- CUSTOM START ---
        my_outerloopcounter = my_outerloopcounter + 1
        #--- CUSTOM END ---
    # completed 2.0 repeats of 'Outerloop'
    print(my_outerloopcounter)
    
    thisExp.nextEntry()
    # the Routine "bidsExport_cva25" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    ignore_list = [
        'participant',
        'session',
        'date',
        'expName',
        'psychopyVersion',
        'OS',
        'frameRate'
    ]
    participant_info = {
        key: thisExp.extraInfo[key]
        for key in thisExp.extraInfo
        if key not in ignore_list
    }
    # write tsv file and update
    try:
        if bids_handler.events:
            bids_handler.writeEvents(participant_info, add_stimuli=True, execute_sidecar=True, generate_hed_metadata=True)
    except Exception as e:
        print(f"[psychopy-bids(settings)] An error occurred when writing BIDS events: {e}")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
