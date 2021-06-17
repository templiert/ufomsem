# As a first approximation, it is assumed that the right half of a sFOV
# is not distorted. This second half is taken as a ground truth
# to compute the scan correction

# To find the scan distortion, a vertical 4 pixels wide band from the translated
# image is fit to the non-translated image.

import os, sys, threading, time
from ij import Macro

from ij import IJ, Macro, ImagePlus, ImageStack
from ij.gui import PointRoi, Roi, Plot
from ij.plugin import MontageMaker
from ij.measure import CurveFitter
from ij.process import ColorProcessor

from java.awt import Rectangle
from java.lang import Math, Runtime, Thread
from java.util.concurrent import Executors
from net.imglib2.img.display.imagej import ImageJFunctions as IL

from register_virtual_stack import Transform_Virtual_Stack_MT

from ini.trakem2 import ControlWindow

from java.util.concurrent.atomic import AtomicInteger

namePlugin = 'computeapply_sc'
ufomsemFolder, ufomsemScriptsFolder = map(
    os.path.normpath,
    (Macro.getOptions()
        .replace('"','')
        .replace(' ','')
        .split('---')))
sys.path.append(ufomsemScriptsFolder)

import fijiCommon as fc
import scan_correction as sc

def compute_round_sc(sc_ref_folder, sc_x_folder):

    # get width,height of first sFOV
    im_path_0 = sc.get_im_paths_from_msem_section_folder(
        sc_x_folder,
        mFOVs=mFOV,
        sFOVs=1)[0]
    im_0 = IJ.openImage(im_path_0)
    w = im_0.getWidth()
    h = im_0.getHeight()

    round_name = os.path.basename(os.path.dirname(sc_x_folder))

    # where to store the scan correction transform
    transform_path = os.path.join(
        scan_corrections_transforms_folder,
        'transform_round_' + round_name + '.txt')

    # where to store the scan distortion measurements
    scan_distortion_measurement_path = os.path.join(
        scan_distortion_folder,
        ('scan_distortion_measurements_x_'
            + '_mFOV_' + str(mFOV)
            + '_' + str(vertical_strip_width) + '_pixels_'
            + round_name
            + '.txt'))

    # path of the overview montage that shows all beam corrections (high and low res)
    fit_montage_highres_path = os.path.join(
        fit_montage_highres_folder,
        'fit_montage_' + str(vertical_strip_width) + '_pixels_' + round_name + '.jpg')
    fit_montage_lowres_path = os.path.join(
        fit_montage_lowres_folder,
        'fit_montage_' + str(vertical_strip_width) + '_pixels_' + round_name + '.jpg')


    # check that the sections for scan correction are not in the exclusion list
    # (sections that have broken through to the substrate
    # typically are in the exclusion list)
    if not any([
        (x in section_exclusion_list)
        for x in ['sc_ref', 'sc_x']]):

        plot_images = []
        x_to_fit, y_to_fit = [], []

        # if the scan distortion measurements have already been done just load them
        if os.path.isfile(scan_distortion_measurement_path):
            mean_a, mean_b, mean_c = sc.get_mean_fit(scan_distortion_measurement_path)
            IJ.log('fit parameters ' + str([mean_a, mean_b, mean_c]))

        else:
            with open(scan_distortion_measurement_path, 'w') as g:

                # looping through all beams (sFOVs)
                for id_sFOV, sFOV in enumerate(sorted(sc.HEXAGONS)):

                    # reference image
                    im_path_1 = sc.get_im_paths_from_msem_section_folder(
                        sc_ref_folder,
                        mFOVs=mFOV,
                        sFOVs=sFOV)[0]

                    # translated image (half of an sFOV to the right)
                    im_path_2 = sc.get_im_paths_from_msem_section_folder(
                        sc_x_folder,
                        mFOVs=mFOV,
                        sFOVs=sFOV)[0]

                    im_1 = IJ.openImage(im_path_1)
                    im_2 = IJ.openImage(im_path_2)

                    # im_1.show()
                    # im_2.show()

                    try:
                        x_points, y_points = sc.compute_scan_distortion(
                            im_1, im_2,
                            vertical_strip_width,
                            extra_width_left, extra_width_right, extra_width,
                            nHighestPeaks)
                    except Exception, e:
                        x_points, y_points = None, None
                        IJ.log('Exception-' + str(e) + '-' + str(sFOV))

                    # # # plot the scan distortion of the current sFOV
                    # # plot = sc.plot_distortion(
                        # # zip(x_points, y_points))
                    # # plot.show()

                    # stack the scan correction measurements in one list to later
                    # fit a single scan correction for all beams.
                    # /!\ assuming that all beams need the same scan correction /!\

                    if x_points and y_points:
                        x_to_fit = x_to_fit + x_points
                        y_to_fit = y_to_fit + y_points

                        # make the fit
                        cv = CurveFitter(x_points, y_points)
                        cv.doFit(CurveFitter.EXP_WITH_OFFSET) # or .POLY3

                        # get plot of fit
                        plot = cv.getPlot()
                        plot.setLimits(0, 1100, -2, 5)
                        plot_images.append(plot.getImagePlus())
                        # plot.show()

                        # write the fit results to file
                        fit_params = cv.getParams()
                        sc.write_fit_params(g, cv, sFOV)
                        IJ.log(
                            ' - sFOV ' + str(sFOV)
                            + ' - fitGoodness ' + str(cv.getFitGoodness())
                            + ' - formula ' + str(cv.getFormula())
                            + ' - fit_params' + str(fit_params))

                        current_a, current_b, current_c = fit_params[:3]

            # Create the 2D visualization of the fit plots
            montage = sc.hexagon_plot_montage(plot_images)
            IJ.save(
                montage,
                fit_montage_highres_path)
            # montage.show()
            small_montage = fc.resize(montage, 0.1)
            IJ.save(
                small_montage,
                fit_montage_lowres_path)
            montage.close()
            small_montage.close()

            mean_a, mean_b, mean_c = sc.get_mean_fit(scan_distortion_measurement_path)
            IJ.log('fit parameters ' + str([mean_a, mean_b, mean_c]))
    else:
        # the forced fit parameters are given in the parameters file
        # these are used when the scan correction sections have broken through the substrate

        IJ.log('Using forced fit parameters')
        mean_a, mean_b, mean_c = forced_fit_params

    # get the scan correction transform
    transform = sc.get_exp_transform(
        w + x_offset, h,
        mesh_x_n, mesh_y_n,
        mean_a, mean_b, mean_c,
        x_offset = x_offset)
    IJ.log('transform: ' + str(transform))

    # save the transform (using the serialize function of a dummy trakem loader)
    pZ, loaderZ, _, _ = fc.getProjectUtils(
                fc.initTrakem(working_folder, 1))
    loaderZ.serialize(transform, transform_path)
    fc.closeProject(pZ)

def apply_section_sc(current_round, current_section, section_folder):
    round_name = os.path.basename(os.path.dirname(section_folder))
    transform_path = os.path.join(
        scan_corrections_transforms_folder,
        'transform_round_' + round_name + '.txt')

    im_paths = sc.get_im_paths_from_msem_section_folder(
        section_folder,
        mFOVs=mFOV)
    IJ.log('im_paths to which SC will be applied: ' + str(im_paths))

    # create the folder (even if autofocus_FAILURE)
    corrected_contrasted_mFOV_folder = os.path.join(
        section_folder.replace(
            root_non_corrected,
            root_corrected_contrasted),
        str(mFOV+1).zfill(6))
    IJ.log('corrected_contrasted_mFOV_folder: ' + str(corrected_contrasted_mFOV_folder))
    if not os.path.exists(corrected_contrasted_mFOV_folder):
        os.makedirs(corrected_contrasted_mFOV_folder)

    if im_paths:
        corrected_contrasted_im_paths = [
            p.replace(root_non_corrected, root_corrected_contrasted)
            for p in im_paths]

        corrected_mFOV_folder = corrected_contrasted_mFOV_folder.replace(
                'corrected_contrasted',
                'corrected')
        if not os.path.exists(corrected_mFOV_folder):
            os.makedirs(corrected_mFOV_folder)

        # load the scan correction transform (with a dummy trakem loader)
        pZ, loaderZ, _, _ = fc.getProjectUtils(
                    fc.initTrakem(working_folder, 1))
        transform = loaderZ.deserialize(transform_path)
        fc.closeProject(pZ)

        # apply scan correction and save transformed images
        sc.apply_sc_to_paths_from_transform(
            im_paths, corrected_contrasted_im_paths,
            transform, x_offset=x_offset,
            clahe=True)

    # update the flag with current round and section
    sc.write_section_round(pipeline_flag_path, current_round, current_section)

    if current_section > 0: # because apply sc to the first 2 sc folders
        fc.terminatePlugin(
            namePlugin,
            ufomsemFolder,
            signalingMessage='kill me')
        time.sleep(5)

#########################################################################
ControlWindow.setGUIEnabled(False)

# reading parameters from text file
ufomsemParameters_path = os.path.join(
    ufomsemFolder,
    'ufomsem_Parameters.txt')
ufomsemParameters = fc.readParameters(ufomsemParameters_path)

n_sections = ufomsemParameters['stitch_align']['n_sections']
# how many sections are created in the trakem
# (probably useless as it is now adding slices on the fly)
n_rounds_max = ufomsemParameters['stitch_align']['n_rounds_max']
# which mFOV to process
mFOV = ufomsemParameters['stitch_align']['mFOV']
# manually entered fit parameters in case calculation
# of scan correction fails
forced_fit_params = ufomsemParameters[namePlugin]['forced_fit_params']
# list of sections that are no longer processed (because advanced break through substrate)
section_exclusion_list = ufomsemParameters['stitch_align']['section_exclusion_list']

# the Zeiss mFOV is 1-based, changing to 0-based
mFOV = mFOV-1

# setting/getting paths of files/folders
root_non_corrected = ufomsemFolder
root_corrected_contrasted = os.path.join(
    os.path.dirname(root_non_corrected),
    'scan_corrected_contrasted_msem')

root_up = os.path.dirname(root_corrected_contrasted)

working_folder = os.path.join(root_up, 'working_folder')
if not os.path.isdir(working_folder):
    os.mkdir(working_folder)

scan_distortion_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'scan_distortion_measurements'))

scan_corrections_transforms_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'scan_correction_transforms'))

fit_montage_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'fit_montages'))

fit_montage_highres_folder = fc.mkdir_p(
    os.path.join(
        fit_montage_folder,
        'fit_montages_highres'))
fit_montage_lowres_folder = fc.mkdir_p(
    os.path.join(
        fit_montage_folder,
        'fit_montages_lowres'))

pipeline_flag_path = os.path.join(
    working_folder,
    'pipeline_flag.txt')

round_folders = sc.get_round_folders(root_non_corrected)
n_rounds = len(round_folders)

# read where we are in the experiment (round, section)
current_round, current_section = sc.init_section_round(
    pipeline_flag_path,
    n_sections)
IJ.log(
    'current_round, current_section'
    + str([current_round, current_section]))

# Number of phase correlation peaks to check with cross-correlation
nHighestPeaks = 5

# get the ROI manager
roi_manager = sc.getRoiManager()

# width of the sliding vertical window
vertical_strip_width = 4

# see drawing. Dimensions of the window in im_1 that
# is also sliding. The window in im_1 is slightly larger than
# the window in im_2
extra_width_left = 6
extra_width_right = 2
extra_width = extra_width_left + extra_width_right

# mesh used to compute the moving least squares transform (landmark-based)
mesh_x_n = 500 # number of points in the mesh on x-axis
mesh_y_n = 30
x_offset = 200 # for applying transform
# x_offset = 200 + int(Math.round(vertical_strip_width/2))

# loop waiting for new sections to be acquired
# checking Zeiss files, whether it is a scan correctection section, etc.
# then deciding on whether to compute and/or apply scan correction
wait_round_flag = True
wait_section_flag = True
if current_section == 0:
    while wait_round_flag:
        time.sleep(1)
        IJ.log('Waiting for the next round to be imaged - Scan correction will be computed')
        round_folders = sc.get_round_folders(root_non_corrected)
        if len(round_folders) > current_round:
            IJ.log('Just found a new round folder - a')
            round_folder = round_folders[current_round]
            n_rounds = n_rounds + 1
            wait_round_flag = False
            IJ.log(
                'a - The new round folder is '
                + str(os.path.basename(round_folder)))
    while wait_section_flag:
        time.sleep(1)
        IJ.log('Waiting for the scan correction sections to be imaged')
        section_folders = sc.subdirs(round_folder)

        try:
            sc_ref_folder = section_folders[
                ['sc_ref' in os.path.basename(f) for f in section_folders].index(True)]
            sc_x_folder = section_folders[
                ['sc_x' in os.path.basename(f) for f in section_folders].index(True)]

            if (sc.is_section_finished(sc_ref_folder)
                and sc.is_section_finished(sc_x_folder)):
                IJ.log('SC images have just been acquired')
                compute_round_sc(sc_ref_folder, sc_x_folder)
                IJ.log('SC computed')
                wait_section_flag = False
        except Exception, e:
            IJ.log('SC images not acquired yet')
            IJ.log('Exception: ' + str(e))

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            IJ.log(str(exc_type) + ';' + str(fname) + ';' + str(exc_tb.tb_lineno))
            # sys.exit()

    IJ.log('This is the first pass, apply SC to the SC folders')
    IJ.log('Apply SC to sc_ref_folder')
    apply_section_sc(current_round, 0, sc_ref_folder)
    IJ.log('Apply SC to sc_x_folder')
    apply_section_sc(current_round, 1, sc_x_folder)
    IJ.log('SC applied to the SC folders')

    # wait_section_flag = True
    # while wait_section_flag:
        # time.sleep(1)
        # section_folders = sc.subdirs(round_folder)
        # if len(section_folders) == 4:
            # section_folder = [f for f in section_folders if 'sc_' not in x][0]
            # apply_section_sc(current_round, current_section, section_folder)

wait_section_flag = True # probably no needed?
round_folder = round_folders[current_round]
IJ.log('the current round folder ' + os.path.basename(round_folder))
while wait_section_flag:
    IJ.log('Waiting for the next section to be imaged')
    time.sleep(1)
    section_folders = sc.subdirs(round_folder)
    IJ.log('section_folders: ' + str(len(section_folders)))
    IJ.log('current_section: ' + str(current_section))
    if len(section_folders) > current_section:
        current_section_folder = section_folders[current_section]
        IJ.log('sc.is_section_finished(current_section_folder): ' + str(sc.is_section_finished(current_section_folder)))
        if sc.is_section_finished(current_section_folder):
            IJ.log('The section just finished')
            apply_section_sc(current_round, current_section, current_section_folder)

