import os, threading, shutil
from jarray import array

from java.io import File
from java.lang import Math, Runtime, Thread
from java.awt import Rectangle
from java.util.concurrent import Executors
from java.util.concurrent.atomic import AtomicInteger

from mpicbg.imglib.image import ImagePlusAdapter
from mpicbg.imglib.algorithm.correlation import CrossCorrelation
from mpicbg.models import TranslationModel2D, RigidModel2D, AffineModel2D
from mpicbg.models import Point, PointMatch
from mpicbg.trakem2.transform import MovingLeastSquaresTransform

from register_virtual_stack import Transform_Virtual_Stack_MT

from ij import IJ, ImagePlus, ImageStack
from ij.gui import PointRoi, Roi, Plot
from ij.plugin import MontageMaker, CanvasResizer
from ij.plugin.frame import RoiManager
from ij.measure import CurveFitter
from ij.process import ColorProcessor

from fiji.selection import Select_Bounding_Box

from bdv.ij import ApplyBigwarpPlugin
from bdv.viewer import Interpolation
from bigwarp.landmarks import LandmarkTableModel

from net.imglib2 import FinalInterval
from net.imglib2.view import Views
from net.imglib2.util import Intervals
from net.imglib2.converter import Converters
from net.imglib2.img.array import ArrayImgFactory
from net.imglib2.algorithm.fft2 import FFTMethods
from net.imglib2.algorithm.fft2 import FFT
from net.imglib2.algorithm.phasecorrelation import PhaseCorrelation2
from net.imglib2.algorithm.phasecorrelation import PhaseCorrelation2Util
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.numeric.complex import ComplexFloatType
from net.imglib2.img.display.imagej import ImageJFunctions as IL

from ij.process import ByteProcessor
# from mpicbg.ij.clahe import Flat
from ini.trakem2.imaging.filters import CLAHE
# from mpicbg.trakem2 import transform as trakem2transform

# order of the 61 beams, line by line left to right. Used to display the topological map of the beams
HEXAGONS = [46,45,44,43,42,47,26,25,24,23,41,48,27,
    12,11,10,22,40,49,28,13,4,3,9,21,39,50,29,14,5,
    1,2,8,20,38,51,30,15,6,7,19,37,61,52,31,16,17,
    18,36,60,53,32,33,34,35,59,54,55,56,57,58]
HEXAGON_SPACINGS = [4,1,1,1,1,7,1,1,1,1,1,5,1,1,1,1,
    1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,3,1,1,1,1,1,1,5,1,1,1,1,1,7,1,1,1,1,4]

def resize(im,factor):
    IJ.run(im,
        'Size...',
        ('width=' + str(int(Math.floor(im.width * factor)))
        + ' height=' + str(int(Math.floor(im.height * factor)))
        + ' average interpolation=Bicubic'))
    return im

def exp_fit(a, b, c, x):
    return a*Math.exp(-b*x) + c

def poly2_fit(x, a, b, c):
    return a + b*x + c*x*x

def poly3_fit(x, a, b, c, d):
    x2 = x*x
    return a + b*x + c*x*x + d *x2*x

def crop(im,roi):
	ip = im.getProcessor()
	ip.setRoi(roi)
	im = ImagePlus(im.getTitle() + '_Cropped', ip.crop())
	return im

def getCC(im1,im2):
	im1, im2 = map(ImagePlusAdapter.wrap, [im1, im2])
	cc = CrossCorrelation(im1, im2)
	cc.process()
	return cc.getR()

def getShiftFromViews(v1, v2, nHighestPeaks, exe=None):
    # Thread pool
    exe_provided = (exe is not None)
    if not exe_provided:
        exe_provided = False
        exe = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    try:
        # PCM: phase correlation matrix
        pcm = PhaseCorrelation2.calculatePCM(v1,
	                                       v2,
	                                       ArrayImgFactory(FloatType()),
	                                       FloatType(),
	                                       ArrayImgFactory(ComplexFloatType()),
	                                       ComplexFloatType(),
	                                       exe)
        # Minimum image overlap to consider, in pixels
        minOverlap = v1.dimension(0) / 10
        # Returns an instance of PhaseCorrelationPeak2
        peak = PhaseCorrelation2.getShift(pcm, v1, v2, nHighestPeaks,
	                                    minOverlap, True, True, exe)
    except Exception, e:
        print 'getShiftFromViews Exception:', e
    finally:
        if not exe_provided:
            exe.shutdown()

    # Register images using the translation (the "shift")
    spshift = peak.getSubpixelShift()
    return spshift.getFloatPosition(0), spshift.getFloatPosition(1)

def getShiftFromImps(imp1, imp2, nHighestPeaks):
    v1 = getViewFromImp(imp1)
    v2 = getViewFromImp(imp2)
    return getShiftFromViews(v1, v2, nHighestPeaks)

def getViewFromImp(imp, r = None):
    # r is a java.awt.rectangle
    im = IL.wrapByte(imp)
    if r is None:
        r = Rectangle(0, 0, imp.getWidth(), imp.getHeight())
    v = Views.zeroMin(Views.interval(im, [r.x, r.y],
                        [r.x + r.width -1, r.y + r.height -1]))
    return v

def getViewFromImglib2Im(img, r):
    v = Views.zeroMin(Views.interval(img, [r.x, r.y],
                        [r.x + r.width -1, r.y + r.height -1]))
    return v

def getFFTParamsFromImps(imp1, imp2, r1 = None, r2 = None):
    if r1:
        v1 = getViewFromImp(
            imp1,
            r1)
    else:
        v1 = IL.wrapByte(imp1)
    if r2:
        v2 = getViewFromImp(
            imp2,
            r2)
    else:
        v2 = IL.wrapByte(imp2)

    extension = array(v1.numDimensions() * [10], 'i')
    extSize = PhaseCorrelation2Util.getExtendedSize(v1, v2, extension)
    paddedDimensions = array(extSize.numDimensions() * [0], 'l')
    fftSize = array(extSize.numDimensions() * [0], 'l')
    return extension, extSize, paddedDimensions, fftSize

def getFFTFromView(v, extension, extSize, paddedDimensions, fftSize, exe=None):
    exe_provided = (exe is not None)
    if not exe_provided:
        exe = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    FFTMethods.dimensionsRealToComplexFast(extSize, paddedDimensions, fftSize)
    fft = ArrayImgFactory(ComplexFloatType()).create(fftSize, ComplexFloatType())
    FFT.realToComplex(Views.interval(PhaseCorrelation2Util.extendImageByFactor(v, extension),
            FFTMethods.paddingIntervalCentered(v, FinalInterval(paddedDimensions))), fft, exe)
    if not exe_provided:
        exe.shutdown()
    return fft

def getShiftFromFFTs(fft1, fft2, v1, v2, minOverlap, nHighestPeaks, exe=None):
    exe_provided = (exe is not None)
    if not exe_provided:
        exe = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())

    pcm = PhaseCorrelation2.calculatePCMInPlace(
        fft1,
        fft2,
        ArrayImgFactory(
            FloatType()),
            FloatType(),
            exe)
    peak = PhaseCorrelation2.getShift(pcm, v1, v2, nHighestPeaks,
                                    minOverlap, True, True, exe)
    if not exe_provided:
        exe.shutdown()

    spshift = peak.getSubpixelShift()
    if spshift is not None:
        return spshift.getFloatPosition(0), spshift.getFloatPosition(1)
    else:
        IJ.log('There is a peak.getSubpixelShift issue')
        return None

def get_im_paths_from_msem_section_folder(section_folder, mFOVs='all', sFOVs='all'):
    # 000005\006_000005_061_2020-12-14T1754552138054.bmp
    im_coordinates_path = os.path.join(
        section_folder,
        'full_image_coordinates.txt')
    if not os.path.isfile(im_coordinates_path):
        return None
    with open(im_coordinates_path, 'r') as f:
        lines = f.readlines()
        sFOV_path_tails = [
            os.path.basename(x.split('\t')[0])
            for x in lines]

    # IJ.log(str(sFOV_path_tails))
    # IJ.log('DEBUG 1 ' + str(mFOVs) + ',' + str(sFOVs) )

    if type(mFOVs) == int:
        mFOVs = [mFOVs]
    if type(sFOVs) == int:
        sFOVs = [sFOVs]
    if sFOVs == 'all':
        sFOVs = range(1,62)

    # IJ.log('DEBUG 2 ' + str(mFOVs) + ',' + str(sFOVs) )
    if mFOVs == 'all':
        sFOV_path_tails = [x for x in sFOV_path_tails
            if (int(x.split('_')[2])-0) in sFOVs]
    else:
        sFOV_path_tails = [x for x in sFOV_path_tails
            if (((int(x.split('_')[2])-0) in sFOVs)
                and ((int(x.split('_')[1])-1) in mFOVs))]
    # IJ.log('DEBUG 3 ' + str(section_folder)
        # + ',' + str(sFOVs)
        # + 'sFOV_path_tails' + str(sFOV_path_tails))

    sFOV_paths = [
        os.path.join(
            section_folder,
            str(int(sFOV_path_tail.split('_')[1])).zfill(6),
            sFOV_path_tail)
        for sFOV_path_tail in sFOV_path_tails]
    # IJ.log('DEBUG 4 ' + str(sFOV_paths))

    return sorted(sFOV_paths)

def getRoiManager():
    roi_manager = RoiManager.getInstance()
    if roi_manager == None:
        roi_manager = RoiManager()
    return roi_manager

def plot_distortion(shifts):
    plot = Plot('Scan distortion', 'spshift', 'offset')
    plot.add(
        'circle',
        [a[0] for a in shifts],
        [a[1] for a in shifts])
    return plot

def write_fit_params(file_handle, cv, sFOV):
    fit_params = cv.getParams()
    file_handle.write(
        'sFOV\t' + str(sFOV) + '\t'
        + 'a\t' + str(fit_params[0]) + '\t'
        + 'b\t' + str(fit_params[1]) + '\t'
        + 'c\t' + str(fit_params[2]) + '\t'
        # + 'formula\t' + str(cv.getFormula()) + '\t'
        + 'fitGoodness\t' + str(cv.getFitGoodness()) + '\n')

def hexagon_plot_montage(plot_images):
    # add all the plots in a stack
    stack_w, stack_h = plot_images[0].getWidth(), plot_images[0].getHeight()
    stack_plot = ImageStack(stack_w, stack_h)

    black_processor = ImagePlus(
        'empty',
        ColorProcessor(stack_w, stack_h)).getProcessor()

    for id, plot_image in enumerate(plot_images):
        for i in range(HEXAGON_SPACINGS[id]):
            stack_plot.addSlice(
                'plot_' + str(id).zfill(2),
                black_processor)
        stack_plot.addSlice(
            'plot_' + str(id).zfill(2),
            plot_image.getProcessor())

    imp = ImagePlus('Plot_stack', stack_plot)
    # return imp

    montageMaker = MontageMaker()
    montage = montageMaker.makeMontage2(
        imp,
        17, 9,
        1, 1,
        imp.getNSlices(), 1, 3, False)

    return montage

def compute_scan_distortion(
    im_1, im_2,
    vertical_strip_width,
    extra_width_left, extra_width_right, extra_width,
    nHighestPeaks):

    # scale_factor = 1
    # im_1 = resize(im_1,scale_factor)
    # im_2 = resize(im_2,scale_factor)
    # extra_width_left = scale_factor*extra_width_left
    # extra_width_right = scale_factor*extra_width_right
    # extra_width = scale_factor*extra_width

    # use the last 10% on the right of im_1 in order to align im_1 to im_2
    start_of_last_10_pct_of_im_1 = Math.floor(0.9 * (im_1.getWidth()))

    # extracting the 10% right vertical band of im_1
    #           _
    # *********|*|
    # *********|*|
    # *********|*|
    # *********|*|
    # *********|*|
    # *********|*|
    #           -

    last_10_pct_of_im_1 = crop(
        im_1,
        Roi(
            start_of_last_10_pct_of_im_1,
            0,
            im_1.getWidth() - start_of_last_10_pct_of_im_1 + 1,
            im_1.getHeight()))

    # last_10_pct_of_im_1.show()

    # calculate the translation offset between im_1 and im_2
    xy_subpixel_shift = getShiftFromImps(
        last_10_pct_of_im_1,
        im_2,
        nHighestPeaks)
    end_of_im_1_in_im_2 = -xy_subpixel_shift[0] + last_10_pct_of_im_1.getWidth()
    IJ.log('end_of_im_1_in_im_2 - ' + str(end_of_im_1_in_im_2))

    img_1 = IL.wrapByte(im_1) #im is an ImagePlus, img is a ImgLib2 image
    img_2 = IL.wrapByte(im_2)

    # get the fft parameters only once at the beginning because they will be te same for all calculations
    extension, extSize, paddedDimensions, fftSize = getFFTParamsFromImps(
        im_1,
        im_2,
        r1 = Rectangle(0, 0, vertical_strip_width + extra_width, im_2.getHeight()),
        r2 = Rectangle(0, 0, vertical_strip_width, im_2.getHeight()))

    # initializing list of subpixelshifts
    spshifts = []

    # sliding the small vertical band (which will be called view v2) pixel by pixel
    # the start of the small vertical band in im_2 is start_of_sliding_band_in_im_2, renamed as "s"

    s_lim = int(Math.ceil(end_of_im_1_in_im_2 - (vertical_strip_width + extra_width_right)))

    # # full range for s
    # s_range = range(s_lim)
    # print 's_lim', s_lim

    # # computing only every two pixels for speedup
    # s_range = range(0, s_lim, 2)

    # # # all pixels are used at the beginning where most of the
    # scan distortion is happening
    # only 1 every n pixels is used afterwards
    boundary = int(s_lim/5)
    s_range = range(0, boundary, 1) + range(boundary, s_lim, 5)
    # print '******** len(s_range)', len(s_range), 100*len(s_range)/s_lim

    for start_of_sliding_band_in_im_2 in s_range:
        s = start_of_sliding_band_in_im_2
        # extract view v1. Instead of looking for where the narrow band v2 fits in the entire im_1,
        # we extract only a narrow band in im_1. We know that v2 will be found in v1.
        # We take the narrow band 3 times wider than v2.

        # the following is clear when looking at the schematic
        start_of_v1_in_im_1 = im_1.getWidth() - (int(round(end_of_im_1_in_im_2)) - s + extra_width_left)

        # the v1 rectangle in which we look for the v2 band is actually quite narrow.
        # we can take it slightly wider than the v2 band: 4 pixels more on the left and 2 pixels more on the right

        r1 = Rectangle(
            start_of_v1_in_im_1,
            0,
            vertical_strip_width + extra_width,
            im_2.getHeight())
        v1 = getViewFromImglib2Im(img_1, r1)
        # if s == 29:
            # crop(im_1, r1).show()
        minOverlap = v1.dimension(0) / 10

        # compute fft_1 of view v1
        fft_1 = getFFTFromView(v1, extension, extSize, paddedDimensions, fftSize)

        if s%50 == 0:
            IJ.log('s-' + str(s).zfill(4))

        # extracting view v2 and computing its fft
        r2 = Rectangle(s, 0, vertical_strip_width, im_2.getHeight())
        v2 = getViewFromImglib2Im(img_2, r2)
        # if s == 29:
            # crop(im_2, r2).show()
        fft_2 = getFFTFromView(v2, extension, extSize, paddedDimensions, fftSize)

        # calculating the spshift between v1 and v2
        # spshift stands for subpixelshift (differs from the "shift" experiments)
        spshift = getShiftFromFFTs(fft_1, fft_2, v1, v2, minOverlap, nHighestPeaks)
        if (spshift is not None):
            spshifts.append([s, spshift[0]])

    x_points = [a[0] for a in spshifts]
    y_points = [extra_width_left - a[1] for a in spshifts]

    # x_points = [int(a[0]/float(scale_factor)) for a in spshifts]
    # y_points = [(extra_width_left - a[1])/float(scale_factor) for a in spshifts]

    return x_points, y_points

def get_exp_transform(
    im_w, im_h,
    x_n, y_n,
    a, b, c, x_offset=0):

    # create mesh
    mesh_x = range(0, im_w, im_w/x_n)
    mesh_y = range(0, im_h, im_h/y_n)

    # normal mesh for im_1
    landmarks_im_1 = [
        [x,y]
        for x in mesh_x
        for y in mesh_y]
    # warped mesh for im_2
    landmarks_im_2 = [
        [x - exp_fit(a, b, c, x-x_offset),
        y]
        for x in mesh_x
        for y in mesh_y]

    # roi2pm = PointRoi()
    # for landmark in landmarks_im_2:
        # roi2pm.addPoint(*landmark)
    # roi_manager.addRoi(roi2pm)

    point_matches = [
        PointMatch(Point(l1), Point(l2))
        for l1,l2 in zip(landmarks_im_1, landmarks_im_2)]

    mlst = MovingLeastSquaresTransform()
    # mlst.setModel(AffineModel2D)
    mlst.setModel(TranslationModel2D)
    mlst.setAlpha(1)
    mlst.setMatches(point_matches)
    return mlst

# should I use a 2-step transformation with high res on the left?
def correct_image(im, transform, transformer, mesh_res=128):
    im_transformed = transformer.applyCoordinateTransform(
        im, transform, mesh_res, True, [0,0])
    return im_transformed

def extend_and_translate(im, x_offset):
    cr = CanvasResizer()
    ip_extended = cr.expandImage(
        im.getProcessor(),
        im.getWidth() + x_offset, im.getHeight(),
        x_offset, 0)
    im_translated = ImagePlus(
        'Translated_' + im.getTitle(),
        ip_extended)
    return im_translated

def auto_crop(im):
    sbb = Select_Bounding_Box()
    crop_box =  sbb.getBoundingBox(
        im.getProcessor(),
        Rectangle(0,0,im.getWidth(),im.getHeight()),
        0)
    sbb.crop(im, crop_box)


def translate_align_ims_to_stack(im_1, im_2):
    # modified from https://syn.mrc-lmb.cam.ac.uk/acardona/fiji-tutorial/#s12
    v1 = getViewFromImp(im_1)
    v2 = getViewFromImp(im_2)
    shift = getShiftFromViews(v1, v2, 10)
    dx = int(shift[0] + 0.5)
    dy = int(shift[1] + 0.5)

    # Top-left and bottom-right corners of the canvas that fits both registered images
    x0 = min(0, dx)
    y0 = min(0, dy)
    x1 = max(v1.dimension(0), v2.dimension(0) + dx)
    y1 = max(v1.dimension(1), v2.dimension(1) + dy)

    canvas_width = x1 - x0
    canvas_height = y1 - y0

    def intoSlice(img, xOffset, yOffset):
      factory = ArrayImgFactory(img.randomAccess().get().createVariable())
      stack_slice = factory.create([canvas_width, canvas_height])
      target = Views.interval(stack_slice, [xOffset, yOffset],
                                           [xOffset + img.dimension(0) -1,
                                            yOffset + img.dimension(1) -1])
      c1 = target.cursor()
      c2 = img.cursor()
      while c1.hasNext():
        c1.next().set(c2.next())
      return stack_slice

    # Re-cut ROIs, this time in RGB rather than just red

    # Insert each into a stack slice
    xOffset1 = 0 if dx >= 0 else abs(dx)
    yOffset1 = 0 if dy >= 0 else abs(dy)
    xOffset2 = 0 if dx <= 0 else dx
    yOffset2 = 0 if dy <= 0 else dy
    slice1 = intoSlice(v1, xOffset1, yOffset1)
    slice2 = intoSlice(v2, xOffset2, yOffset2)

    stack = Views.stack([slice1, slice2])
    im_stack = IL.wrap(stack, 'registered with phase correlation')
    return im_stack

def remove_first_last_columns(im, i,j):
    ip = im.getProcessor()
    roi = Roi(
        i, 0,
        im.getWidth() -i - j, im.getHeight())
    ip.setRoi(roi)
    im = ImagePlus(im.getTitle(), ip.crop())
    return im

def remove_borders(im, left, right, top, bottom):
    ip = im.getProcessor()
    roi = Roi(
        left,
        top,
        im.getWidth() -left - right,
        im.getHeight() - top - bottom)
    ip.setRoi(roi)
    im = ImagePlus(im.getTitle(), ip.crop())
    return im

def apply_transform_to_paths_parallel(
    atom,
    source_paths, target_paths,
    transform, transformer, x_offset, clahe):

    while atom.get() < len(source_paths):
        k = atom.getAndIncrement()
        if (k < len(source_paths)):
            source_path = source_paths[k]
            target_path = target_paths[k]

            if True and (not os.path.isfile(target_path)):
                IJ.log('Processing image ' + str(k))
                # IJ.log('source_path ' + source_path)
                # IJ.log('target_path ' + target_path)

                im = IJ.openImage(source_path)
                im_translated = extend_and_translate(im, x_offset) #provide canvasresizer?
                corrected_im = correct_image(
                    im_translated,
                    transform, transformer,
                    mesh_res=32)
                auto_crop(corrected_im)

                corrected_cropped_im = remove_borders(
                    corrected_im.duplicate(),
                    left=1, right=2,
                    top=0, bottom=2)

                non_contrasted_target_path = target_path.replace('_contrasted', '')
                IJ.save(corrected_cropped_im, non_contrasted_target_path)

                local_contrast(corrected_im)
                corrected_im = remove_borders(
                    corrected_im,
                    left=1, right=2,
                    top=0, bottom=2)

                # IJ.run(
                    # corrected_im,
                    # 'Enhance Local Contrast (CLAHE)',
                    # 'blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate)')
                IJ.save(corrected_im, target_path)
                im.close()
                corrected_im.close()

def apply_sc_to_paths(
    source_paths, target_paths,
    mesh_x_n, mesh_y_n,
    a, b, c, x_offset=200,
    clahe=True):

    im_0 = IJ.openImage(source_paths[0])
    w = im_0.getWidth()
    h = im_0.getHeight()
    im_0.close()
    transform = get_exp_transform(
        w + x_offset, h,
        mesh_x_n, mesh_y_n,
        a, b, c,
        x_offset = x_offset)

    transformer = Transform_Virtual_Stack_MT()
    atom = AtomicInteger(0)
    threads = []
    for p in range(32):
        thread = threading.Thread(
            group = None,
            target = apply_transform_to_paths_parallel,
            args = (atom,
                    source_paths, target_paths,
                    transform, transformer, x_offset, clahe))
        threads.append(thread)
        thread.start()
        IJ.log('Thread ' + str(p) + ' started')
    for thread in threads:
        thread.join()

    # apply_transform_to_paths(
        # source_paths, target_paths,
        # transform, transformer, x_offset)

def apply_sc_to_paths_from_transform(source_paths, target_paths,
    transform, x_offset=200,
    clahe=True):

    # with open(transform_path, 'r') as f:
        # transform_string = f.readline()
    transformer = Transform_Virtual_Stack_MT()
    # transform = transformer.readCoordinateTransform(transform_path)
    IJ.log('The transform ' + str(transform))

    atom = AtomicInteger(0)
    threads = []
    for p in range(32):
        thread = threading.Thread(
            group = None,
            target = apply_transform_to_paths_parallel,
            args = (atom,
                    source_paths, target_paths,
                    transform, transformer, x_offset, clahe))
        threads.append(thread)
        thread.start()
        IJ.log('Thread ' + str(p) + ' started')
    for thread in threads:
        thread.join()
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def duplicate_folder_tree(source, target):
    try:
        shutil.copytree(
            source,
            target,
            ignore=ig_f)
    except Exception as e:
        print 'copytree exception', e
    # for dirpath, dirnames, filenames in os.walk(source):
        # structure = os.path.join(target, os.path.relpath(dirpath, source))
        # if not os.path.isdir(structure):
            # os.mkdir(structure)
        # else:
            # print("Folder does already exits!")

def subdirs(folder):
    subdirs = []
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            subdirs.append(os.path.join(root, dir) )
        break
    return sorted(subdirs)

def local_contrast(im, block = 127, histobins = 256, maxslope = 3):
    clahe = CLAHE(True, block, histobins, maxslope)
    ip = im.getProcessor()
    clahe.process(ip)

    # ipMaskCLAHE = ByteProcessor(im.getWidth(),im.getHeight())
    # ipMaskCLAHE.threshold(-1)
    # bitDepth = im.getBitDepth()
    # if bitDepth == 8:
        # maxDisp = Math.pow(2,8) - 1
    # else:
        # maxDisp = Math.pow(2,12) - 1

    # ip = im.getProcessor()
    # ip.setMinAndMax(0,maxDisp)
    # if bitDepth == 8:
        # ip.applyLut()
    # Flat.getFastInstance().run(im, block, histobins, maxslope, ipMaskCLAHE, False)
    # del ipMaskCLAHE
    # return im

def get_exp_log_lines(experiment_folder):
    exp_log_path = os.path.join(
        experiment_folder,
        'experiment_log.txt')
    if not os.path.isfile(exp_log_path):
        return None
    else:
        with open(exp_log_path, 'r') as f:
            lines = f.readlines()
        return lines

def get_stop_time(experiment_folder):
    lines = get_exp_log_lines(experiment_folder)
    if not lines:
        return None
    if 'stop experiment' in lines[-1]:
        stop_time = datetime.strptime(
            lines[-1].split(':')[0][:-1],
            '%Y-%m-%dT%H%M%S%f') + timedelta(hours=-5)
        return stop_time
    else:
        return None

def is_round_finished(round_folder):
    lines = get_exp_log_lines(round_folder)
    if not lines:
        return False
    return ('stop experiment' in lines[-1])

def is_section_finished(section_folder):
    # # rare example
    # processing finished, 2021-02-12T0609222278573
    # errors, 0
    # region finished, 2021-02-12T0609222278573

    flagfile_path = os.path.join(
        section_folder,
        'flagfile.txt')
    if not os.path.isfile(flagfile_path):
        IJ.log('No section flag file yet')
        return False
    with open(flagfile_path, 'r') as f:
        lines = f.readlines()
    IJ.log('lines: ' + str(lines))
    return any(['processing finished' in line for line in lines])

def read_current_section_round(pipeline_flag_path):
    with open(pipeline_flag_path, 'r') as f:
        last_round, last_section = map(int,f.readline().split('\t'))
    return last_round, last_section


def init_section_round(pipeline_flag_path, n_sections):
    if not os.path.isfile(pipeline_flag_path):
        with open(pipeline_flag_path, 'w') as f:
            f.write(str(n_sections-1) +'\t-1')
        last_round, last_section= -1, n_sections -1
    else:
        last_round, last_section = read_current_section_round(pipeline_flag_path)

    if last_section == n_sections-1:
        current_round = last_round + 1
        current_section = 0
    else:
        current_round = last_round
        current_section = last_section + 1
    IJ.log('Initialized round section ' + str([current_round, current_section]))
    return current_round, current_section

def write_section_round(pipeline_flag_path, current_round, current_section):
    IJ.log('Writing section round ' + str([current_round, current_section]))
    with open(pipeline_flag_path, 'w') as f:
        f.write(str(current_round) + '\t' + str(current_section))

def get_round_folders(f):
    round_folders = subdirs(f)
    round_folders = [
        f for f in round_folders
        if (not 'Logs' in os.path.basename(f))]
    return round_folders

def get_mean_fit(result_path):
    all_a = []
    all_b = []
    all_c = []

    # IJ.log('result_path: ' + result_path)
    results = read_fit_results(result_path)
    # IJ.log('***results***' + str(results))
    if results:
        a = [line[1] for line in results]
        b = [line[2] for line in results]
        c = [line[3] for line in results]
        all_a.append(a)
        all_b.append(b)
        all_c.append(c)
    else:
        all_a.append(None)
        all_b.append(None)
        all_c.append(None)

    IJ.log('all_a ' + str(all_a))
    IJ.log('all_b ' + str(all_b))
    IJ.log('all_c ' + str(all_c))

    mean_a = [sum(filter(None,a))/len(filter(None,a)) for a in all_a if a][0]
    mean_b = [sum(filter(None,b))/len(filter(None,b)) for b in all_b if b][0]
    mean_c = [sum(filter(None,c))/len(filter(None,c)) for c in all_c if c][0]

    return mean_a, mean_b, mean_c

def read_fit_results(p):
    if os.path.isfile(p):
        with open(p, 'r') as f:
            lines = f.readlines()
            lines = [line.split('\t') for line in lines]
            lines = [[line[1], line[3], line[5], line[7]] for line in lines]
            lines = [[float(x) for x in line] for line in lines]
        return lines
    else:
        IJ.log('no result path')
        return None

def save_gif_from_stack(stack, duration, gif_path):
    # stack = fc.stackFromPaths(export_paths)
    stack_size = stack.getStackSize()
    if stack_size>1:
        IJ.run(
            stack,
            'Animation Options...',
            'speed=' + str(stack_size/float(duration)) + ' first=1 last=' + str(stack_size))
    IJ.saveAs(stack, 'gif', gif_path)
    # else:
        # IJ.save(stack, gif_path.replace('.gif', '.tif'))
    # stack.close()
