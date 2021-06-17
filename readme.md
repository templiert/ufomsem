# ufomsem
This script keeps starting Fiji plugins one after the other in an infinite loop while an acquisition experiment is running. The script also forcefully terminates Fiji between each plugin to minimize failures.

The 3 scripts in the pipeline are 'computeapply_sc', 'stitch_align', 'check_section'.

The base experiment folder is named msem. The scripts create folders: scan_corrected_msem, scan_corrected_contrasted_msem and the main working_folder with more subfolders.

# computeapply_sc

The first script 'computeapply_sc' is checking for newly acquired sections. The imaging experiment starts with 3 sections used for scan correction named sc_ref, sc_x, sc_y.

- If the last acquired section is sc_x, then the script computes the scan correction for the imaging round and applies it to sc_ref, sc_x.
- If the last acquired section is not sc_ref,sc_x, then it applies the computed scan correction to the section.

The computed fits are stored in working_folder\scan_distortion_measurements.

Only the mFOV defined in the parameters file is processed.
The raw sFOVs are scan corrected and saved with the same original folder structure. It creates both scan corrected sFOVs (scan_corrected_msem) and scan corrected CLAHE sFOVs (scan_corrected_contrasted_msem).

# stitch_align

It creates trakem projects and inserts the sFOVs according to the Zeiss initial positioning.
Simple phase correlation stitching is applied, then affine 3d alignment.

# check_section

It checks several metrics (mean, stdev, focus score, etc.).
If the last acquired section is the last section of the imaging round, then the script assembles a summary gif and plots and it sends a summary email.

![Naming convention](scan_correction_naming.jpg?raw=true "Naming convention")