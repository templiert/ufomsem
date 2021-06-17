import os, time, sys
from ij import IJ, Macro
import java
from java.lang import Runtime
from java.awt import Rectangle
from java.awt.geom import AffineTransform
from java.util import HashSet
from ini.trakem2 import Project, ControlWindow
from ini.trakem2.display import Patch, Display
from ini.trakem2.imaging import StitchingTEM
from ini.trakem2.imaging.StitchingTEM import PhaseCorrelationParam
from mpicbg.trakem2.align import RegularizedAffineLayerAlignment

###
namePlugin = 'stitch_align'
ufomsemFolder, ufomsemScriptsFolder = map(
    os.path.normpath,
    (Macro.getOptions()
        .replace('"','')
        .replace(' ','')
        .split('---')))
sys.path.append(ufomsemScriptsFolder)

import fijiCommon as fc
import scan_correction as sc

def custom_resize_display(project, l):
    # manually resizing one trakem layer
    # called because sometimes an unexplainable expansion
    # of the display happens
    p, loader, layerset, nLayers = fc.getProjectUtils(project)
    layer = layerset.getLayers().get(l)
    patches = layer.getDisplayables(Patch)
    minX = min([patch.getX() for patch in patches])
    minY = min([patch.getY() for patch in patches])
    if (minX != 0) or (minY != 0):
        for patch in patches:
            patch.setLocation(
                patch.getX() - minX,
                patch.getY() - minY)
            patch.updateBucket()

def align_layer_to_previous(layerset, l):
    # alignment parameters
    siftP = {}
    siftP[0] = ['SIFTfdBins', 8]
    siftP[1] = ['SIFTfdSize', 8]
    siftP[2] = ['SIFTinitialSigma', 1.6]
    siftP[3] = ['SIFTmaxOctaveSize', 2000]
    siftP[4] = ['SIFTminOctaveSize', 1000]
    siftP[5] = ['SIFTsteps', 1]
    siftP[6] = ['clearCache', True]
    siftP[7] = ['maxNumThreadsSift', Runtime.getRuntime().availableProcessors()]
    siftP[8] = ['rod', 0.92] # Closest/next closest neighbour distance ratio
    siftP[9] = ['desiredModelIndex', 3]
    siftP[10] = ['expectedModelIndex', 3]
    siftP[11] = ['identityTolerance', 1]
    siftP[12] = ['lambdaa', 0.1]
    siftP[13] = ['maxEpsilon', 200]
    siftP[14] = ['maxIterationsOptimize', 1000]
    siftP[15] = ['maxNumFailures',3]
    siftP[16] = ['maxNumNeighbors',neighbors] # should be changed to 5 for larger stacks
    siftP[17] = ['maxNumThreads', Runtime.getRuntime().availableProcessors()]
    siftP[18] = ['maxPlateauwidthOptimize', 200]
    siftP[19] = ['minInlierRatio', 0]
    siftP[20] = ['minNumInliers', 12]
    siftP[21] = ['multipleHypotheses', False]
    siftP[22] = ['widestSetOnly', False]
    siftP[23] = ['regularize', True]
    siftP[24] = ['regularizerIndex', 1]
    siftP[25] = ['rejectIdentity', False]
    siftP[26] = ['visualize', False]

    siftParams = RegularizedAffineLayerAlignment.Param(
        *[ a[1] for a in [siftP[i] for i in range(len(siftP))]])

    layBound = layerset.get2DBounds()
    IJ.log('layBound ' + str(layBound))
    box = Rectangle(layBound.width, layBound.height)

    # fixing the existing layers
    fixedLayers = HashSet()
    for i in range(l):
        fixedLayers.add(layerset.getLayers().get(i))

    emptyLayers = HashSet()
    propagateTransformBefore = False
    propagateTransformAfter = False

    layerRange = layerset.getLayers(l-1,l)
    IJ.log('layerRange: ' + str(layerRange))


    # aligning the new slice to the current stack
    # this alignment is done twice with different parameters
    # reason is it seems to improve reliability of the alignment
    # trying to continue if there is a failure
    try:
        try:
            RegularizedAffineLayerAlignment().exec(
                siftParams,
                layerRange,
                fixedLayers,
                emptyLayers,
                box,
                propagateTransformBefore,
                propagateTransformAfter,
                None)
        except java.lang.Exception, err:
            IJ.log('*Warning* : alignment has failed: ' + str(err))
    except Exception, e:
        IJ.log('*Warning* : alignment has failed: python' + str(e))

    # use the center rectangle to align, instead of the whole mFOV with the black borders
    last_layer_bb = layerset.getLayers().get(l-1).getMinimalBoundingBox(Patch)
    box = Rectangle (
        last_layer_bb.x + 5400,
        last_layer_bb.y + 6500,
        4000,
        4000)


    # redoing the alignment, this time with a different maxOctaveSize
    siftP[3] = ['SIFTmaxOctaveSize', 1500]
    siftParams = RegularizedAffineLayerAlignment.Param(*[ a[1] for a in [siftP[i] for i in range(len(siftP))] ])

    fixedLayers = HashSet()
    for i in range(l):
        fixedLayers.add(layerset.getLayers().get(i))

    emptyLayers = HashSet()
    propagateTransformBefore = False
    propagateTransformAfter = False

    layerRange = layerset.getLayers(l-1,l)
    IJ.log('layerRange: ' + str(layerRange))

    try:
        try:
            RegularizedAffineLayerAlignment().exec(
                siftParams,
                layerRange,
                fixedLayers,
                emptyLayers,
                box,
                propagateTransformBefore,
                propagateTransformAfter,
                None)
        except java.lang.Exception, err:
            IJ.log('*Warning* : alignment has failed in second step: ' + str(err))
    except Exception, e:
        IJ.log('Warning, python exception alignment failed: ' + str(e))

def write_trakem_msem_import_from_file(
    input_positions_path, # output from Zeiss with coordinates of sFOVs
    import_file_path, # file for trakem for importing sFOVs
    root_non_corrected, root_corrected,
    mFOV, layer):

    line_count = 0
    with open(import_file_path, 'w') as g:
        with open(input_positions_path, 'r') as f:
            lines = f.readlines()

        # manually offset the sFOV positions because I cannot control
        # how the trakem layer resizing works and some drift sometimes happens.
        # easiest is resizing here at the start, simply finding the bounding box

        # update the relative path to absolute, and update the layer number
        minX, minY = 10e10, 10e10
        for line in lines:
            splitLine = line.split('\t')
            minX = min(minX, float(splitLine[-3]))
            minY = min(minY, float(splitLine[-2]))
        for line in lines:
            splitLine = line.split('\t')
            if splitLine[0][:6] == str(mFOV).zfill(6):
                line_count +=1
                splitLine[0] = os.path.join(
                    os.path.dirname(input_positions_path),
                    splitLine[0])
                splitLine[0] = splitLine[0].replace(
                    root_non_corrected,
                    root_corrected)
                splitLine[-3] = str(float(splitLine[-3]) - minX)
                splitLine[-2] = str(float(splitLine[-2]) - minY)

                splitLine[-1] = str(layer)
                newLine = '\t'.join(splitLine) + '\n'
                g.write(newLine)
    return line_count

def write_trakem_msem_import(
    stitched_positions_path,
    import_file_path,
    root_non_corrected, root_corrected,
    mFOV, layer):

    n_lines = write_trakem_msem_import_from_file(
        stitched_positions_path,
        import_file_path,
        root_non_corrected, root_corrected,
        mFOV, layer)

    IJ.log('n_lines: ' + str(n_lines))

    if n_lines != 61:
        IJ.log('WARNING: Zeiss initial stitching failed -> using unstitched file positions')
        input_positions_path = os.path.join(
            os.path.dirname(stitched_positions_path),
            'full_image_coordinates.txt')

        write_trakem_msem_import_from_file(
            input_positions_path,
            import_file_path,
            root_non_corrected, root_corrected,
            mFOV, layer)

def import_images_from_file(layerset, loader, import_file_path):
    # trakem loading of sFOVs from description file
    task = loader.importImages(
        layerset.getLayers().get(0),
        import_file_path, '\t', 1, 1, False, 1, 0)
    task.join()

def get_first_empty_layer(layerset):
    layers = layerset.getLayers()
    for l, layer in enumerate(layers):
        if layer.getNDisplayables() == 0:
            return l
    return None

def duplicate_modify_trakem(
    # util to change the path of the source images
    # in a trakem project
    source_path, target_path,
    source_text, target_text):

    with open(source_path, 'r') as f:
        filedata = f.read()
    filedata = filedata.replace(source_text, target_text)
    with open(target_path, 'w') as g:
        g.write(filedata)

def duplicate_modify_trakems(project_path):
    # create Trakems for non-contrasted tissue, and for Ken without the M:\hess prefix
    duplicate_modify_trakem(
        project_path,
        project_path.replace(
            'trakems_contrasted',
            'trakems_non_contrasted'),
        'corrected_contrasted',
        'corrected')
    duplicate_modify_trakem(
        project_path,
        project_path.replace(
            'trakems_contrasted',
            'trakems_contrasted_mount'),
        'M:/hess/TT',
        'M:/TT')
    duplicate_modify_trakem(
        project_path.replace(
            'trakems_contrasted',
            'trakems_non_contrasted'),
        project_path.replace(
            'trakems_contrasted',
            'trakems_non_contrasted_mount'),
        'M:/hess/TT',
        'M:/TT')

def stitch_align_section(current_round, current_section, section_folder):
    section_name = os.path.basename(section_folder)[4:] # Zeiss section naming is always: 002_OurSectionName
    project_path = os.path.join(
        trakems_contrasted_folder,
        section_name + '.xml')
    IJ.log('project_path - ' + project_path)

    # load project, create if does not exist
    if not os.path.isfile(project_path):
        IJ.log('No current project: creating it')
        project, loader, layerset, _ = fc.getProjectUtils(
            fc.initTrakem(
                trakems_contrasted_folder,
                n_rounds_max))
        project.saveAs(project_path, True)
    else:
        IJ.log('Ongoing project found')
        while True:
            try:
                project, loader, layerset, _ = fc.openTrakemProject(project_path)
                break
            except java.lang.Exception, err:
                IJ.log('Error loading the trakem project, trying again: ' + str(err))
                time.sleep(1)

    stitching_params = PhaseCorrelationParam(
        1, 0.08,
        False, False,
        2.5, 0.3)

    # the Zeiss "stitched" positions: no stitching is made
    # so these are stage/beam coordinates
    stitched_positions_path = os.path.join(
        section_folder,
        section_name + '_stitched_imagepositions.txt')
    IJ.log('stitched_positions_path ' + str(stitched_positions_path))

    import_file_path = os.path.join(working_folder, 'import_file.txt')

    # # # # empty the layer before importing in case of a restart
    # # # layer = layerset.getLayers().get(current_round)
    # # # displayables = layer.getDisplayableList()

    # # # if len(displayables) !=0:
        # # # IJ.log('*** Warning : the layer is not empty. It must be a restart.')
        # # # layer.remove(False)

    # if relative coordinates, then update to absolute coordinates
    if os.path.isfile(stitched_positions_path):
        write_trakem_msem_import(
            stitched_positions_path,
            import_file_path,
            root_non_corrected, root_corrected,
            mFOV, current_round)

        import_images_from_file(layerset, loader, import_file_path)
        IJ.log('All sFOVs imported into layer ' + str(current_round))

        # resize display
        custom_resize_display(project, current_round)
        fc.resizeDisplay(layerset)

        # simple stitching with phaseCorrelation
        patches = layerset.getLayers().get(current_round).getDisplayables(Patch)
        try:
            try:
                StitchingTEM().montageWithPhaseCorrelation(
                    patches,
                    stitching_params)
            except java.lang.Exception, err:
                IJ.log('*Warning* : stitching has failed: ' + str(err))
        except Exception, e:
            IJ.log('Warning: stitching failed python: ' + str(e))

        if current_round > 0:
            if (not (section_name in section_exclusion_list)
                and not (section_name in ['sc_ref', 'sc_x', 'sc_y', 'pc_x', 'pc_y']) ):
                # to avoid failures and to speed up things, no alignment for excluded sections
                # and for "scan correction sections"

                align_layer_to_previous(layerset, current_round)
                IJ.log('Section ' + section_name + ' is not in exclusion list')
            else:
                IJ.log('Section ' + section_name + ' is in exclusion list')

    else:
        print 'WARNING no Zeiss initial positions file'

    # because the loader-import triggers layerset resizing, always use first layer as reference
    layer_0_bb = layerset.getLayers().get(0).getMinimalBoundingBox(Patch)
    layer_0_bb.x = layer_0_bb.x
    layer_0_bb.y = layer_0_bb.y

    # updated position of center rectangle relative to first layer
    updated_rectangle = Rectangle(
        gif_rectangle[0] + layer_0_bb.x,
        gif_rectangle[1] + layer_0_bb.y,
        gif_rectangle[2],
        gif_rectangle[3])

    # exporting downsampled stack for small gif in summary email
    export_stack = fc.exportFlat_get_stack(
        project,
        gif_scale_factor,
        bitDepth = 8,
        layers = range(current_round+1),
        roi = updated_rectangle)

    gif_path = os.path.join(
        animated_gifs_folder,
        'animated_gif_' + section_name + '_' + str(current_round).zfill(3) + '.gif')
    previous_gif_path = os.path.join(
        animated_gifs_folder,
        'animated_gif_' + section_name + '_' + str(current_round-1).zfill(3) + '.gif')

    sc.save_gif_from_stack(export_stack, gif_duration, gif_path)

    # export stack to file (high-res)
    high_res_contrasted_export_folder = os.path.join(
        high_res_contrasted_exports_folder,
        section_name)

    if os.path.isdir(high_res_contrasted_export_folder):
        export_range = [current_round]
    else:
        os.mkdir(high_res_contrasted_export_folder)
        export_range = range(current_round+1)
    # high_res_export_folder = fc.mkdir_p(
        # os.path.join(
            # high_res_exports_folder,
            # section_name))

    if highres_export_rectangle == 'all':
        export_roi = Rectangle(
            layer_0_bb.x,
            layer_0_bb.y,
            17000, 14650)

            # the roi should not change across sections
    else:
        export_roi = Rectangle(
            layer_0_bb.x + highres_export_rectangle[0],
            layer_0_bb.y + highres_export_rectangle[1],
            highres_export_rectangle[2],
            highres_export_rectangle[3])


    if not (section_name in ['sc_ref', 'sc_x', 'sc_y', 'pc_x', 'pc_y']):
        fc.exportFlat(
            project,
            high_res_contrasted_export_folder,
            1,
            baseName=section_name,
            bitDepth=8,
            layers = export_range,
            roi = export_roi)

    # fc.resizeDisplay(layerset) # removed: it makes a different treatment to the contrasted stack and to the non-contrasted stack

    project.save()
    fc.closeProject(project)
    duplicate_modify_trakems(project_path)

    # high res export NON-contrasted

    non_contrasted_project_path = project_path.replace(
        '_contrasted',
        '_non_contrasted')

    while True:
        try:
            project, loader, layerset, _ = fc.openTrakemProject(non_contrasted_project_path)
            break
        except java.lang.Exception, err:
            IJ.log('Error loading the non-contrasted trakem project, trying again: ' + str(err))
            time.sleep(1)

    high_res_non_contrasted_export_folder = os.path.join(
        high_res_non_contrasted_exports_folder,
        section_name)

    if os.path.isdir(high_res_non_contrasted_export_folder):
        export_range = [current_round]
    else:
        os.mkdir(high_res_non_contrasted_export_folder)
        export_range = range(current_round+1)
    # high_res_export_folder = fc.mkdir_p(
        # os.path.join(
            # high_res_exports_folder,
            # section_name))

    if not (section_name in ['sc_ref', 'sc_x', 'sc_y', 'pc_x', 'pc_y']):
        fc.exportFlat(
            project,
            high_res_non_contrasted_export_folder,
            1,
            baseName=section_name,
            bitDepth=8,
            layers = export_range,
            roi = export_roi)

    project.save()
    fc.closeProject(project)

    if os.path.isfile(previous_gif_path):
        os.remove(previous_gif_path)
        IJ.log('Removing previous gif: ' + str(previous_gif_path))


# this script is starting without information about the experiment status
ControlWindow.setGUIEnabled(False)

# reading parameters from file
ufomsemParameters_path = os.path.join(
    ufomsemFolder,
    'ufomsem_Parameters.txt')
ufomsemParameters = fc.readParameters(ufomsemParameters_path)

n_sections = ufomsemParameters[namePlugin]['n_sections']
n_rounds_max = ufomsemParameters[namePlugin]['n_rounds_max']
mFOV = ufomsemParameters[namePlugin]['mFOV']

SIFTmaxOctaveSize = ufomsemParameters[namePlugin]['SIFTmaxOctaveSize']
SIFTminOctaveSize = ufomsemParameters[namePlugin]['SIFTminOctaveSize']
SIFTsteps = ufomsemParameters[namePlugin]['SIFTsteps']
neighbors = ufomsemParameters[namePlugin]['neighbors']

gif_scale_factor = ufomsemParameters[namePlugin]['gif_scale_factor']
gif_rectangle = ufomsemParameters[namePlugin]['gif_rectangle']
gif_duration = ufomsemParameters[namePlugin]['gif_duration']

highres_export_rectangle = ufomsemParameters[namePlugin]['highres_export_rectangle']
section_exclusion_list = ufomsemParameters[namePlugin]['section_exclusion_list']
IJ.log('section_exclusion_list: ' + str(section_exclusion_list))

# setting up file/folder paths
root_non_corrected = ufomsemFolder
root_corrected = os.path.join(
    os.path.dirname(root_non_corrected),
    'scan_corrected_contrasted_msem')
root_up = os.path.dirname(root_corrected)

working_folder = os.path.join(root_up, 'working_folder')
if not os.path.isdir(working_folder):
    os.mkdir(working_folder)

animated_gifs_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'animated_gif_single_sFOVs'))

trakem_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'trakems'))

trakems_contrasted_folder = fc.mkdir_p(
    os.path.join(
        trakem_folder,
        'trakems_contrasted'))
trakems_non_contrasted_folder = fc.mkdir_p(
    os.path.join(
        trakem_folder,
        'trakems_non_contrasted'))
trakems_contrasted_mount_folder = fc.mkdir_p(
    os.path.join(
        trakem_folder,
        'trakems_contrasted_mount'))
trakems_non_contrasted_mount_folder = fc.mkdir_p(
    os.path.join(
        trakem_folder,
        'trakems_non_contrasted_mount'))

high_res_exports_folder = fc.mkdir_p(
    os.path.join(
        working_folder,
        'high_res_exports'))

high_res_contrasted_exports_folder = fc.mkdir_p(
    os.path.join(
        high_res_exports_folder,
        'contrasted'))

high_res_non_contrasted_exports_folder = fc.mkdir_p(
    os.path.join(
        high_res_exports_folder,
        'non_contrasted'))

pipeline_flag_path = os.path.join(
    working_folder,
    'pipeline_flag.txt')

# reads the experiment state from a simple text flag file
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

if current_section == 1:
    stitch_align_section(
        current_round,
        current_section - 1,
        section_folders[current_section - 1])
    IJ.log('This part is executed only if current_section==1, '
        + 'meaning that it must be rerun to process the second SC folder')

stitch_align_section(
    current_round,
    current_section + 1,
    current_section_folder)

# the external python orchestrator will terminate this Fiji instance
fc.terminatePlugin(
    namePlugin,
    ufomsemFolder,
    signalingMessage='kill me')


#sc.is_round_finished(round_folder)
