# ufomsem
This script keeps starting Fiji plugins and subsequently terminating Fiji in an infinite loop. The scripts are:
1.computeapply sc
2.stitch align
3.check section

The base experiment folder is named msem. The scripts create folders: scan_corrected_msem, and scan_corrected_contrasted_msem and the main working_folder.

# Computeapply_sc

The first script 1.computeapply is checking for new sections acquired. The imaging experiment starts with 3 sections used for scan correction named sc_ref, sc_x, sc_y.

If the last acquired section is sc_x, then the script computes the scan correction for the imaging round and applies it to sc_ref, sc_x.
If the last acquired section is not sc_ref,sc_x, then it applies the computed scan correction to the section.

The computed exponential fits are stored in working_folder\scan_distortion_measurements.

It is processing the mFOV defined in the parameters file.
It is transforming the raw sFOVs and reproducing the same folder structure. It creates both scan corrected sFOVs (scan_corrected_msem) and scan corrected CLAHE sFOVs (scan_corrected_contrasted_msem).

# stitch_align

It creates trakem projects and inserts the sFOVs according to the Zeiss initial positioning.
Phase correlation stitching is applied, then affine 3d alignment is performed.

# check section

It checks several metrics (mean, stdev, focus score, etc.).
It the last acquired section is the last section of the imaging round, then the script assembles a summary gif and plots and sends a summary by email.

![Naming convention](scan_correction_naming.jpg?raw=true "Naming convention")