# orchestrator script that launches all scripts for ufomsem

import tkinter
from tkinter import filedialog
import os, sys, time
import shutil, psutil
import subprocess
import signal
from subprocess import call, Popen
import argparse
import platform

def getDirectory(text):
    root = tkinter.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=text) + os.sep
    path = os.path.join(path, '')
    return path

def askFile(*args):
    root = tkinter.Tk()
    root.withdraw()
    if len(args) == 1:
        path =  filedialog.askopenfilename(title=args[0])
    else:
        path = filedialog.askopenfilename(title=args[0], initialdir=args[1])
    return path

def whereAmI():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, '')

def whereIs(item, itemType, displayText, ufomsemScriptsFolder, isNew):
    storedItemPath = os.path.join(
        ufomsemScriptsFolder ,
        'whereIs' + item + '.txt')
    try:
        if isNew:
            raise IOError
        with open(storedItemPath , 'r') as f:
            itemPath = f.readline()
            itemPath = itemPath.replace('\n', '').replace(' ', '')
            itemPath = os.path.join(itemPath)
        if itemType == 'file' and not os.path.isfile(itemPath):
            raise IOError
    except IOError:
        print('I do not know where ', item, ' is')
        if itemType == 'file':
            try:
                itemPath = askFile(displayText)
            except Exception as e:
                print('Please create yourself the files whereIsFiji6.txt,'
                + 'whereIsFiji8.txt, whereIsufomsemFolder.txt in the ufomsem folder.'
                + ' Each file contains the folder or fiji executable location in one line')
                sys.exit()
        elif itemType == 'folder':
            itemPath = getDirectory(displayText)
        with open(storedItemPath, 'w') as f:
            f.write(itemPath)
    return itemPath

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='',
        help = '"new" (to trigger a dialog to enter the path of '
            + 'the main ufomsem folder) OR The path to the parent '
            + 'folder that contains all the ufomsem data')
    parser.add_argument('-f', default='',
        help = '"new" (to trigger a dialog to enter the path to '
            + 'the Fiji executable) OR The path to the Fiji executable')
    args = parser.parse_args()

    ufomsemScriptsFolder = whereAmI()
    # get the fiji Path
    if args.p == '' or args.p == 'new':
        fiji8Path = whereIs('Fiji8', 'file',
            'Please select the *** JAVA 8 *** Fiji',
            ufomsemScriptsFolder, args.p == 'new')
    else:
        fiji8Path = os.path.normPath(args.p) # broken because of 2 fiji

    if args.p == '' or args.p == 'new':
        fiji6Path = whereIs('Fiji6', 'file',
            'Please select the *** JAVA 6 *** Fiji',
            ufomsemScriptsFolder, args.p == 'new')
    else:
        fiji6Path = os.path.normPath(args.p) # broken because of 2 fiji

    # plugins folder based on fiji path
    fijiPluginsFolders = [
        os.path.join(
            os.path.split(fijiPath)[0],
            'plugins','')
            for fijiPath in [fiji8Path, fiji6Path]]

    # copy all the scripts into the plugins folders of Fiji
    for fijiPluginsFolder in fijiPluginsFolders:
        for root, dirs, files in os.walk(ufomsemScriptsFolder):
            for file in filter(lambda x: x.endswith('.py'), files):
                    shutil.copy(os.path.join(root, file), fijiPluginsFolder)

    # get the ufomsemFolder path
    if args.f == '' or args.f == 'new':
        ufomsemFolder = whereIs(
            'ufomsemFolder',
            'folder',
            'Please select the ufomsem folder',
            ufomsemScriptsFolder, args.f == 'new')
    else:
        ufomsemFolder = os.path.join(os.path.normPath(args.f),'')

    # If the ufomsem_Parameters file is not there, then add the standard one from the repo
    ufomsemParamPath = os.path.join(
        ufomsemFolder,
        'ufomsem_Parameters.txt')
    if not os.path.isfile(ufomsemParamPath): # ufomsem_Parameters is not in the data folder
        shutil.copy(
            os.path.join(
                ufomsemScriptsFolder,
                'ufomsem_Parameters.txt'),
            ufomsemFolder)

    return ufomsemFolder, fiji8Path, fiji6Path

def cleanPathForFijiCall(path):
# the path here is provided as an argument to a Fiji script. The path has to be handled differently if it is a folder or a file path so that Fiji understands it well.
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        path = os.path.join(path, '')
    path = path.replace(os.sep, 2*os.sep)
    return path

def runFijiScript(plugin):
    plugin = plugin[0]
    fijiFlag = plugin[1]

    repeat = True

    signalingPath = os.path.join(
        ufomsemFolder,
        'signalingFile_' + plugin.replace(' ', '_') + '.txt')
    print('signalingPath', signalingPath)

    plugin = "'" + plugin + "'"

    # passing as argument both ufomsemFolder and ufomsemScriptFolder
    arguments = (
        cleanPathForFijiCall(ufomsemFolder)
        + '---'
        + cleanPathForFijiCall(whereAmI()))

    while repeat:
        print(
            'running plugin ',
            plugin,
            ' : ',
            str(time.strftime('%Y%m%d-%H%M%S')))

        # print(' with arguments ', arguments)
        # command = fijiPath + ' -eval ' + '"run(' + plugin + ",'" + arguments  + "'"

        if fijiFlag == 0:
            fijiPath = fiji8Path
        else:
            fijiPath = fiji6Path

        command = (
            fijiPath
            + ' -eval '
            + '"run('
            + plugin
            + ",'"
            + arguments
            + "'"
            + ')"')
        print('command', command)

        if platform.system() == 'Linux':
            p = subprocess.Popen(
                command,
                shell=True,
                preexec_fn = os.setsid) # do not use stdout = ... otherwise it hangs
        else:
            p = subprocess.Popen(
                command,
                shell=True) # do not use stdout = ... otherwise it hangs

            # result = subprocess.call(command, shell=True)

        # print('subprocess', p)

        waitingForPlugin = True
        while waitingForPlugin:
            # print('waitingForPlugin')
            if os.path.isfile(signalingPath):
                #time.sleep(2)
                with open(signalingPath, 'r') as f:
                    line = f.readlines()[0]
                    print('line ------', line)

                    ########################
                    ##### Killing Loop #####
                    killed = False
                    while not killed:

                        print('Killing ImageJ...')
                        if platform.system() == 'Linux':
                            print('os.getpgid(p.pid)', os.getpgid(p.pid), p.pid)
                            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                        else:
                            a = subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])

                        killed = True
                        for proc in psutil.process_iter():
                            # if 'ImageJ' in proc.name() and proc.ppid()==p.pid:
                            try:
                                if proc.ppid()==p.pid and 'ImageJ' in proc.name():
                                    print('ImageJ still alive', proc)
                                    # print('ImageJ ',
                                        # proc.pid,
                                        # proc.ppid(),
                                        # '\n',
                                        # proc.cmdline(),
                                        # '\n',
                                        # command,
                                        # '\n',
                                        # command == proc.cmdline(),
                                        # proc.create_time(),
                                        # p.pid)
                                    killed = False
                            except Exception as e:
                                print('Process no longer exists: it has been killed in the meantime --- ', e)

                        time.sleep(0.2)
                        # if a ==0:
                            # killed = True
                            # print('process successfully killed')
                        # else:
                            # print('process not killed yet')
                    ##### End Killing Loop #####
                    ############################

                    if line == 'kill me':
                        print(plugin , ' has run successfully: ', str(time.strftime('%Y%m%d-%H%M%S')))
                        repeat = False

                    elif line == 'kill me and rerun me':
                        print(plugin , ' has run successfully and needs to be rerun ', str(time.strftime('%Y%m%d-%H%M%S')))

                    else:
                        print('********************* ERROR')
                print('signalingPath from ufomsem', signalingPath)
                os.remove(signalingPath)
                time.sleep(2)
                waitingForPlugin = False
            time.sleep(1)

        # # # if result == 0:
            # # # print 'result',result
            # # # print plugin , ' has run successfully: ', str(time.strftime('%Y%m%d-%H%M%S'))
            # # # repeat = False
        # # # elif result == 2:
            # # # print plugin , ' has run successfully and needs to be rerun ', str(time.strftime('%Y%m%d-%H%M%S'))
        # # # else:
            # # # print plugin, ' has failed'
            # # # sys.exit(1)

def findFilesFromTags(folder,tags):
    filePaths = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if (all(map(lambda x:x in filename,tags)) == True):
                path = os.path.join(dirpath, filename)
                # return [path] # simply returns the first occurence
                filePaths.append(path)
    return filePaths

#############################################################
# Script starts here
#############################################################

ufomsemFolder, fiji8Path, fiji6Path = init()

pipeline = [
    ['computeapply sc', 0], # 0,1 for whether Fiji Java 6 or Fiji java 8 to be used (historical reasons, one thing was crashing in trakem java 8)
    ['stitch align', 0],
    ['check section', 0],
    ]

# runFijiScript(pipeline[0])
# runFijiScript(pipeline[1])
# runFijiScript(pipeline[2])

while True:
    for step in pipeline:
        runFijiScript(step)