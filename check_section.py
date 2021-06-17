import os, time, sys
from ij import IJ, Macro, ImagePlus, ImageStack
from java.lang import Runtime
from java.awt import Rectangle
from java.awt.geom import AffineTransform
from java.util import HashSet
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch
from ini.trakem2.imaging import StitchingTEM
from ini.trakem2.imaging.StitchingTEM import PhaseCorrelationParam
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment
from ij.plugin import MontageMaker

from ij.plugin.filter import GaussianBlur as Blur
from ij.plugin import ImageCalculator

import base64

def encode(t):
    return base64.b64encode(t).encode('utf-8')
def decode(t):
    return base64.b64decode(t).decode('utf-8')

def blur(im,sigma):
	blur = Blur()
	ip = im.getProcessor()
	blur.blurGaussian(
        ip,
        sigma,
        sigma,
        0.0005)
	im = ImagePlus(
        os.path.splitext(im.getTitle())[0] + '_Blur',
        ip)
	return im

def focus_score(im, g_1, g_2):
    im_g_1 = blur(im.duplicate(),g_1)
    im_g_2 = blur(im.duplicate(),g_2)
    ic = ImageCalculator()
    diff_1 = im_g_1.duplicate()
    ic.run('subtract',diff_1, im_g_2)
    diff_2 = im_g_2.duplicate()
    ic.run('subtract', diff_2,im_g_1)
    ic.run('add', diff_1,diff_2)
    ic.run('multiply', diff_1,diff_1)
    mean = diff_1.getStatistics().mean
    im_g_1.close()
    im_g_2.close()
    diff_1.close()
    diff_2.close()
    return mean

def get_mean_stdev(im):
    stats = im.getStatistics()
    return stats.mean, stats.stdDev

def get_pixel_connectivity_from_section_folder(section_folder):
    pixco_path = os.path.join(
        section_folder,
        'pixelConnectivityResults.txt')
    if not os.path.isfile(pixco_path):
        return -999
    with open(pixco_path, 'r') as f:
        lines = f.readlines()
    averages = [
        float(line.split('\t')[2]) for line in lines[1:]]
    average = round(
        sum(averages)/float(len(averages)))
    return average

def get_n_support_points(experiment_folder):
    lines = sc.get_exp_log_lines(experiment_folder)
    if not lines:
        return None
    n_support_points = []

    for line in lines:
        try:
            points = int(
                line.split('Support Points: ')[1]
                .split(';')[0])
            n_support_points.append(points)
        except Exception, e:
            pass
    return n_support_points

def get_successful_AFs(experiment_folder):
    lines = sc.get_exp_log_lines(experiment_folder)
    if not lines:
        return None
    AFs = []

    for line in lines:
        try:
            AF = int(
                line.split('successful AutoFocus: ')[1]
                .split(';')[0])
            AFs.append(AF)
        except Exception, e:
            pass
    return AFs

def get_successful_stigs(experiment_folder):
    lines = sc.get_exp_log_lines(experiment_folder)
    if not lines:
        return None
    stigs = []

    for line in lines:
        try:
            stig = int(
                line.split('successful AutoStig: ')[1])
            stigs.append(stig)
        except Exception, e:
            pass
    IJ.log('stigs: ' + str(stigs))
    return stigs

def send_email_python(
    sender,
    recipients,
    subject,
    message,
    im_path,
    working_folder):

    command = ' '.join([
        'python',
        r'D:\TT\tt_janelia\msem\send_email_with_graphs_python.py',
        encode(sender),
        encode(','.join(recipients)),
        encode(subject),
        encode(message),
        encode(im_path),
        encode(working_folder)])

    IJ.log('Command: ' + str(command))
    # concordePath + ' -o ' + solutionPath + ' ' + tsplibPath

    process = Runtime.getRuntime().exec(command)
    process.waitFor()

def make_montage_gif(gifs_folder):
    # make animated gifs of all imaging rounds and sections
    # the sections are shown in a 2d montage
    # and the gif animation goes through the imaging rounds

    # there are already existing gifs for each section
    # this function opens all the gifs and combines them
    # into a montaged gif
    gif_paths = fc.naturalSort([
        os.path.join(gifs_folder, name)
        for name in os.listdir(gifs_folder)
        if name.endswith('.gif')])
    gifs = [IJ.openImage(p) for p in gif_paths]
    IJ.log('gif_paths ' + str(gif_paths))
    n_slices = gifs[0].getStackSize()
    w = gifs[0].getWidth()
    h = gifs[0].getHeight()
    montageMaker = MontageMaker()
    montages = []
    for id_slice in range(n_slices):
        ips = [
            gif.getStack().getProcessor(id_slice + 1)
            for gif in gifs]

        # section_names = [
         # os.path.basename(gif_path).split('_')[2]
         # for gif_path in gif_paths]

        section_names = [
            os.path.basename(gif_path).split('_')[2] + '_' + os.path.basename(gif_path).split('_')[3]
            for gif_path in gif_paths]

        stack = ImageStack(w, h)
        for section_name, ip in zip(section_names, ips):
            stack.addSlice(section_name, ip)
        montage_stack = ImagePlus('Montage_' + str(id_slice), stack)
        # montage.setDimensions(1, 1, len(ips))
        montage = montageMaker.makeMontage2(
            montage_stack,
            montage_grid[0], montage_grid[1],
            1,
            1, montage_stack.getNSlices(), 1,
            3, True)

        montages.append(montage)

    w = montages[0].getWidth()
    h = montages[0].getHeight()
    stack_montage_for_gif = ImageStack(w,h)
    for montage in montages:
        stack_montage_for_gif.addSlice(montage.getProcessor())
    im_stack_montage_for_gif = ImagePlus('Montage_gif_round_' + str(current_round), stack_montage_for_gif)
    sc.save_gif_from_stack(im_stack_montage_for_gif, gif_duration, montage_gif_path)
    return montage_gif_path

def check_section_quality(
    current_round,
    current_section,
    current_section_folder):

    # this function computes the following three metrics
    # in all the sFOVs of one mFOV
    sFOV_focus_scores = []
    intensity_means = []
    stdevs = []

    im_paths = sc.get_im_paths_from_msem_section_folder(
        current_section_folder,
        mFOVs=focus_mFOVs,
        sFOVs=focus_sFOVs)
    if im_paths:
        im_paths = [p.replace(root_non_corrected, root_corrected) for p in im_paths]
        for im_path in im_paths:
            IJ.log('im_path: ' + im_path)
            im = IJ.openImage(im_path)
            sFOV_focus_scores.append(focus_score(im.duplicate(), 1, 4))
            sFOV_mean, sFOV_stdev = get_mean_stdev(im)
            intensity_means.append(sFOV_mean)
            stdevs.append(sFOV_stdev)
            im.close()

        section_focus = sum(sFOV_focus_scores)/len(sFOV_focus_scores)
        section_mean = sum(intensity_means)/len(intensity_means)
        section_stdev = sum(stdevs)/len(stdevs)
    else:
        section_focus = -999
        section_mean = -999
        section_stdev = -999

    IJ.log(
        'section_focus, section_mean, section_stdev ' + str([section_focus, section_mean, section_stdev])
        + ' current_round ' + str(current_round)
        + ' current_section ' + str(current_section))
    with open(section_metrics_path, 'w') as f:
        f.write('\t'.join(map(str,[
            section_focus,
            section_mean,
            section_stdev])))

def read_errors_from_section_folder(section_folder):
    # reading errors from Zeiss file
    flagfile_path = os.path.join(
        section_folder,
        'flagfile.txt')
    with open(flagfile_path, 'r') as f:
        lines = f.readlines()

    error_lines = [
        line for line in lines
        if 'errors' in line]
    if len(error_lines) == 1:
        error_line = error_lines[0].split(',')[1:]
    else:
        error_line = []

    errors = {
    'Autofocus': sum([int(x.split('101#')[1]) if ('101#' in x) else 0 for x in error_line]),
    'Autostig': sum([int(x.split('102#')[1]) if ('102#' in x) else 0 for x in error_line]),
    'Beam2Fiber': sum([int(x.split('103#')[1]) if ('103#' in x) else 0 for x in error_line]),
    'File': sum([int(x.split('201#')[1]) if ('201#' in x) else 0 for x in error_line]),
    'Processing': sum([int(x.split('301#')[1]) if ('301#' in x) else 0 for x in error_line])
    }
    return errors

# tools for html formating of summary email
def red(t):
    return colorize(t, 'red')
def green(t):
    return colorize(t, 'green')

def colorize(t, color):
    if color == 'green':
        c = '#008000'
    if color == 'red':
        c = '#FF0000'
    colorized_text = '<span style="color: ' + c + '">' + t + '</span>'
    return colorized_text

def add_sections_to_exclusion_list(section_names):
    # handling of how sections are referred to
    global section_exclusion_list

    new_section_names = [
        name for name in section_names
        if not (name in section_exclusion_list)]

    if not new_section_names:
        return
    new_string = ','.join(new_section_names)

    IJ.log('section_exclusion_list :' + str(section_exclusion_list))

    with open(ufomsemParameters_path, 'r') as f :
        filedata = f.read()
    if section_exclusion_list != '':
        filedata = filedata.replace(
            str(section_exclusion_list[-1]) + ']',
            str(section_exclusion_list[-1]) + ',' + new_string + ']')
    else:
        filedata = filedata.replace(
            'section_exclusion_list = []',
            'section_exclusion_list = [' + new_string + ']')
        filedata = filedata.replace(
            'section_exclusion_list =[]',
            'section_exclusion_list = [' + new_string + ']')
        filedata = filedata.replace(
            'section_exclusion_list= []',
            'section_exclusion_list = [' + new_string + ']')
        filedata = filedata.replace(
            'section_exclusion_list=[]',
            'section_exclusion_list = [' + new_string + ']')

    with open(ufomsemParameters_path, 'w') as g:
        g.write(filedata)

###
namePlugin = 'check_section'
ufomsemFolder, ufomsemScriptsFolder = map(
    os.path.normpath,
    (Macro.getOptions()
        .replace('"','')
        .replace(' ','')
        .split('---')))
sys.path.append(ufomsemScriptsFolder)

import fijiCommon as fc
import scan_correction as sc

ControlWindow.setGUIEnabled(False)
ufomsemParameters_path = os.path.join(
    ufomsemFolder,
    'ufomsem_Parameters.txt')
ufomsemParameters = fc.readParameters(ufomsemParameters_path)

n_sections = ufomsemParameters['stitch_align']['n_sections']
n_rounds_max = ufomsemParameters['stitch_align']['n_rounds_max']
mFOV = ufomsemParameters['stitch_align']['mFOV']
focus_sFOVs = ufomsemParameters[namePlugin]['focus_sFOVs']
focus_mFOVs = ufomsemParameters[namePlugin]['focus_mFOVs']
intensity_mean_termination_threshold = ufomsemParameters[namePlugin]['intensity_mean_termination_threshold']
montage_grid = ufomsemParameters[namePlugin]['montage_grid']

gif_duration = ufomsemParameters['stitch_align']['gif_duration']
section_exclusion_list = ufomsemParameters['stitch_align']['section_exclusion_list']
if section_exclusion_list != '':
    if type(section_exclusion_list) != list:
        section_exclusion_list = [section_exclusion_list]
IJ.log('section_exclusion_list --- ' + str(section_exclusion_list))
IJ.log('type(section_exclusion_list) --- ' + str(type(section_exclusion_list)))
IJ.log('section_exclusion_list==None --- ' + str(section_exclusion_list==None))
IJ.log('section_exclusion_list=="" --- ' + str(section_exclusion_list==''))

sender = 'templiert@hhmi.org'
# recipients = [
    # 'templiert@hhmi.org']
# recipients = [
    # 'templiert@hhmi.org',
    # 'd.peale.1@mindspring.com',
    # 'hayworthk@janelia.hhmi.org',
    # 'hessh@janelia.hhmi.org']
recipients = [
    'templiert@hhmi.org',
    'd.peale.1@mindspring.com',
    'hayworthk@janelia.hhmi.org']
# recipients = [
    # 'templiert@hhmi.org',
    # 'hayworthk@janelia.hhmi.org']

# mFOV might be a list or single value
# changing to 0-based
if type(focus_mFOVs) == int:
    focus_mFOVs = focus_mFOVs -1
else:
    focus_mFOVs = [f-1 for f in focus_mFOVs]

root_non_corrected = ufomsemFolder
root_corrected = os.path.join(
    os.path.dirname(root_non_corrected),
    'scan_corrected_contrasted_msem')
root_up = os.path.dirname(root_corrected)

working_folder = os.path.join(root_up, 'working_folder')

pipeline_flag_path = os.path.join(
    working_folder,
    'pipeline_flag.txt')

current_round, current_section = sc.read_current_section_round(pipeline_flag_path)

IJ.log(
    'current_round, current_section'
    + str([current_round, current_section]))

round_folders = sc.get_round_folders(root_non_corrected)
n_rounds = len(round_folders)

round_folder = round_folders[current_round]
IJ.log('the current round folder ' + os.path.basename(round_folder))
section_folders = sc.subdirs(round_folder)
current_section_folder = section_folders[current_section]
IJ.log('Current section folder: ' + str(current_section_folder))
IJ.log('read_errors_from_section_folder(section_folder): ' + str(read_errors_from_section_folder(current_section_folder)))

gifs_folder = os.path.join(
    working_folder,
    'animated_gif_single_sFOVs')

metrics_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'section_metrics'))

section_metrics_folder = fc.mkdir_p(
    os.path.join(
        metrics_folder,
        'metrics_round_' + str(current_round).zfill(3)))

montage_gif_rounds_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'montage_gif_rounds'))

if current_section == 1:
    IJ.log('This part is executed only if current_section==1, '
        + 'meaning that it must be rerun to process the second SC folder')
    section_metrics_path = os.path.join(
        section_metrics_folder,
        ('section_metrics_' + str(current_round).zfill(3)
            + '_' + str(current_section - 1).zfill(3) + '.txt'))
    check_section_quality(
        current_round,
        current_section - 1,
        section_folders[current_section - 1])

section_metrics_path = os.path.join(
    section_metrics_folder,
    ('section_metrics_' + str(current_round).zfill(3)
        + '_' + str(current_section).zfill(3) + '.txt'))

check_section_quality(
    current_round,
    current_section,
    current_section_folder)

if current_section == n_sections-1:
    # end of the imaging round
    IJ.log('Sending summary email')

    montage_gif_path = os.path.join(
        montage_gif_rounds_folder,
        'montage_gif_round_' + str(current_round) + '.gif')

    fit_montage_lowres_path = os.path.join(
        working_folder,
        'fit_montages',
        'fit_montages_lowres',
        'fit_montage_4_pixels_' + os.path.basename(round_folder) + '.jpg')

    focus_scores = []
    intensity_means = []
    intensity_stdevs = []
    pixcos = []
    AFs = get_successful_AFs(round_folder)
    stigs = get_successful_stigs(round_folder)
    n_support_points = get_n_support_points(round_folder)

    errors_Autofocus = []
    errors_Autostig = []
    errors_Beam2Fiber = []
    errors_File = []
    errors_Processing = []
    section_names_to_exclude = []

    email_text = '(If you do not want to receive these automated emails let me know) <br> <br>'
    email_text += ('Section..'
        + 'PixCo..'
        + 'AFs....'
        + 'Stigs....'
        + 'Focus..'
        + 'Mean...'
        + 'Stdev..'
        + 'ErrAF..'
        + 'ErrAS..'
        + 'ErrB2F..'
        + 'ErrFile..'
        + 'ErrProc..'
        + '<br><br>')

    for id_section in range(n_sections):
        section_folder = section_folders[id_section]
        section_name = os.path.basename(section_folder).split('_')[1] + '_' + os.path.basename(section_folder).split('_')[2]
        section_metrics_path = os.path.join(
            section_metrics_folder,
            ('section_metrics_' + str(current_round).zfill(3)
                + '_' + str(id_section).zfill(3) + '.txt'))
        with open(section_metrics_path, 'r') as f:
            focus, intensity_mean, intensity_stdev = map(float, f.readline().split('\t'))

            IJ.log('[intensity_mean, intensity_stdev]: ' + str([intensity_mean, intensity_stdev, section_name]))
            if ((intensity_mean > intensity_mean_termination_threshold) and (intensity_stdev < 19)):
                # or (intensity_mean==-999)
                IJ.log('Adding to exclusion list: ' + section_name)
                section_names_to_exclude.append(section_name)

            focus_scores.append(focus)
            intensity_means.append(intensity_mean)
            intensity_stdevs.append(intensity_stdev)
        pixcos.append(
            get_pixel_connectivity_from_section_folder(
                section_folder))

        errors_section = read_errors_from_section_folder(section_folder)
        errors_Autofocus.append(errors_section['Autofocus'])
        errors_Autostig.append(errors_section['Autostig'])
        errors_Beam2Fiber.append(errors_section['Beam2Fiber'])
        errors_File.append(errors_section['File'])
        errors_Processing.append(errors_section['Processing'])

        error_Autofocus_text = colorize(
            str(errors_Autofocus[id_section]),
            'red' if errors_Autofocus[id_section] else 'green')
        error_Autostig_text = colorize(
            str(errors_Autostig[id_section]),
            'red' if errors_Autostig[id_section] else 'green')
        error_Beam2Fiber_text = colorize(
            str(errors_Beam2Fiber[id_section]),
            'red' if errors_Beam2Fiber[id_section] else 'green')
        error_File_text = colorize(
            str(errors_File[id_section]),
            'red' if errors_File[id_section] else 'green')
        error_Processing_text = colorize(
            str(errors_Processing[id_section]),
            'red' if errors_Processing[id_section] else 'green')

        email_text +=  (
            section_name
            + (9-len(section_name)) * '.'

            + str(int(round(pixcos[id_section])))
            + (7-len(str(int(round(pixcos[id_section]))))) * '.'

            + str(AFs[id_section]) + os.sep + str(n_support_points[id_section])
            + (7-len(str(AFs[id_section]) + os.sep + str(n_support_points[id_section]))) * '.'

            + str(stigs[id_section]) + os.sep + str(n_support_points[id_section])
            + (9-len(str(stigs[id_section]) + os.sep + str(n_support_points[id_section]))) * '.'

            + str(int(round(focus_scores[id_section])))
            + (7-len(str(int(round(focus_scores[id_section]))))) * '.'

            + str(int(round(intensity_means[id_section])))
            + (7-len(str(int(round(intensity_means[id_section]))))) * '.'

            + str(int(round(intensity_stdevs[id_section])))
            + (7-len(str(int(round(intensity_stdevs[id_section]))))) * '.'

            + error_Autofocus_text
            + (7 - len(str(errors_Autofocus[id_section]))) * '.'

            + error_Autostig_text
            + (7 - len(str(errors_Autostig[id_section]))) * '.'

            + error_Beam2Fiber_text
            + (8 - len(str(errors_Beam2Fiber[id_section]))) * '.'

            + error_File_text
            + (9 - len(str(errors_File[id_section]))) * '.'

            + error_Processing_text
            + (9 - len(str(errors_Processing[id_section]))) * '.'

            + '<br>')

    add_sections_to_exclusion_list(section_names_to_exclude)
    email_text += ('<br> <br>The following sections seem to be finished and are not processed for alignment any more: <br><br>'
        + str(fc.naturalSort(fc.readParameters(ufomsemParameters_path)['stitch_align']['section_exclusion_list'])) + '<br> <br>')

    send_email_python(
        sender,
        recipients,
        'Summary of imaging round #' + str(current_round) + ': ' + os.path.basename(round_folder),
        email_text,
        ','.join([
            make_montage_gif(gifs_folder),
            fit_montage_lowres_path]),
        working_folder)

    IJ.log('email sent')
    # time.sleep(60)

fc.terminatePlugin(
    namePlugin,
    ufomsemFolder,
    signalingMessage='kill me')
