# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Processing of dynamic field maps from complex-valued multi-echo BOLD data."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import Split as FSLSplit
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from sdcflows.interfaces.fmap import PhaseMap2rads, ROMEO
from sdcflows.utils.misc import calculate_diffs, medic_automask

INPUT_FIELDS = ("magnitude", "phase", "metadata")


def init_medic_wf(
    n_volumes,
    echo_times,
    automask,
    automask_dilation,
    omp_nthreads=1,
    sloppy=False,
    debug=False,
    name="medic_wf",
):
    """
    Create the PEPOLAR field estimation workflow based on FSL's ``topup``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_topup_wf
            wf = init_topup_wf()

    Parameters
    ----------
    sloppy : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    magnitude : :obj:`list` of :obj:`str`
        A list of echo-wise magnitude EPI files that will be fed into MEDIC.
    phase : :obj:`list` of :obj:`str`
        A list of echo-wise phase EPI files that will be fed into MEDIC.
    metadata : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the metadata corresponding to each file
        in ``in_data``.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.
    fmap_mask : :obj:`str`
        The path of mask corresponding to the ``fmap_ref`` output.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "jacobians",
                "xfms",
                "out_warps",
                "method",
            ],
        ),
        name="outputnode",
    )
    outputnode.inputs.method = "MEDIC"

    n_echoes = len(echo_times)

    # Convert phase to radians (-pi to pi)
    phase2rad = pe.MapNode(
        PhaseMap2rads(),
        iterfield=["in_file"],
        name="phase2rad",
    )
    workflow.connect([(inputnode, phase2rad, [("phase", "in_file")])])

    # Split phase data into single-frame, multi-echo 4D files
    group_phase_across_echoes = pe.MapNode(
        niu.Merge(numinputs=n_echoes),
        iterfield=[f"in{i + 1}" for i in range(n_echoes)],
        name="group_phase_across_echoes",
    )
    group_mag_across_echoes = pe.MapNode(
        niu.Merge(numinputs=n_echoes),
        iterfield=[f"in{i + 1}" for i in range(n_echoes)],
        name="group_mag_across_echoes",
    )
    for i_echo in range(n_echoes):
        select_phase_echo = pe.Node(
            niu.Select(index=i_echo),
            name=f"select_phase_echo_{i_echo:02d}",
        )
        workflow.connect([(phase2rad, select_phase_echo, [("out_file", "inlist")])])

        split_phase = pe.Node(
            FSLSplit(dimension="t"),
            name=f"split_phase_{i_echo:02d}",
        )
        workflow.connect([
            (select_phase_echo, split_phase, [("out", "in_file")]),
            (split_phase, group_phase_across_echoes, [("out_files", f"in{i_echo + 1}")]),
        ])  # fmt:skip

        # Split magnitude data into single-frame, multi-echo 4D files
        select_mag_echo = pe.Node(
            niu.Select(index=i_echo),
            name=f"select_mag_echo_{i_echo:02d}",
        )
        workflow.connect([(inputnode, select_mag_echo, [("magnitude", "inlist")])])

        split_mag = pe.Node(
            FSLSplit(dimension="t"),
            name=f"split_mag_{i_echo:02d}",
        )
        workflow.connect([
            (select_mag_echo, split_mag, [("out", "in_file")]),
            (split_mag, group_mag_across_echoes, [("out_files", f"in{i_echo + 1}")]),
        ])  # fmt:skip

    for i_volume in range(n_volumes):
        process_volume_wf = init_process_volume_wf(
            echo_times,
            automask,
            automask_dilation,
            name=f"process_volume_{i_volume:02d}_wf",
        )

        select_phase_volume = pe.Node(
            niu.Select(index=i_volume),
            name=f"select_phase_volume_{i_volume:02d}",
        )
        select_mag_volume = pe.Node(
            niu.Select(index=i_volume),
            name=f"select_mag_volume_{i_volume:02d}",
        )
        workflow.connect([
            (group_phase_across_echoes, select_phase_volume, [("out", "inlist")]),
            (group_mag_across_echoes, select_mag_volume, [("out", "inlist")]),
            (select_phase_volume, process_volume_wf, [("out", "inputnode.phase")]),
            (select_mag_volume, process_volume_wf, [("out", "inputnode.magnitude")]),
        ])  # fmt:skip

    # Re-combine into echo-wise time series

    # Check temporal consistency of phase unwrapping

    # Compute field maps

    # Apply SVD filter to field maps

    return workflow


def init_process_volume_wf(
    echo_times,
    automask,
    automask_dilation,
    name="process_volume_wf",
):
    """Create a workflow to process a single volume of multi-echo data.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_process_volume_wf

            wf = init_process_volume_wf(
                echo_times=[0.015, 0.030, 0.045, 0.06],
                automask=True,
                automask_dilation=3,
            )   # doctest: +SKIP

    Parameters
    ----------
    echo_times : :obj:`list` of :obj:`float`
        The echo times of the multi-echo data.
    automask : :obj:`bool`
        Whether to automatically generate a mask for the fieldmap.
    automask_dilation : :obj:`int`
        The number of voxels by which to dilate the automatically generated mask.

    Inputs
    ------
    magnitude : :obj:`str`
        The magnitude EPI file that will be fed into MEDIC.
    phase : :obj:`str`
        The phase EPI file that will be fed into MEDIC.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.
    fmap_mask : :obj:`str`
        The path of mask corresponding to the ``fmap_ref`` output.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["magnitude", "phase"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "fmap_ref", "fmap_mask"]),
        name="outputnode",
    )

    mask_buffer = pe.Node(
        niu.IdentityInterface(fields=["mask_file", "masksum_file"]),
        name="mask_buffer",
    )
    if automask:
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive
        # occasionally and the voxel quality mask can get extra voxels that are not brain,
        # but is noisy so we combine the two masks to get a better mask

        # Use ROMEO's voxel-quality command
        voxqual = pe.MapNode(
            ROMEO(write_quality=True, echo_times=echo_times),
            name="voxqual",
            iterfield=["in_file"],
        )
        workflow.connect([
            (inputnode, voxqual, [
                ("magnitude", "mag_file"),
                ("phase", "phase_file"),
            ]),
        ])  # fmt:skip

        # Then use skimage's otsu thresholding to get a mask
        # and do a bunch of other stuff
        automask_medic = pe.Node(
            niu.Function(
                input_names=["mag_file", "voxel_quality", "echo_times", "automask_dilation"],
                output_names=["mask_file", "masksum_file"],
                function=medic_automask,
            ),
            name="automask_medic",
        )
        automask_medic.inputs.echo_times = echo_times
        automask_medic.inputs.automask_dilation = automask_dilation
        workflow.connect([
            (inputnode, automask_medic, [("magnitude", "mag_file")]),
            (voxqual, automask_medic, [("quality_file", "voxel_quality")]),
            (automask_medic, mask_buffer, [
                ("mask_file", "mask_file"),
                ("masksum_file", "masksum_file"),
            ]),
        ])  # fmt:skip
    else:
        mask_buffer.inputs.mask_file = "NULL"
        mask_buffer.inputs.masksum_file = "NULL"

    # Do MCPC-3D-S algo to compute phase offset
    create_diffs = pe.MapNode(
        niu.Function(
            input_names=["mag_file", "phase_file"],
            output_names=["phase_offset", "unwrapped_diff", "phase_modified"],
            function=calculate_diffs,
        ),
        iterfield=["mag_file", "phase_file"],
        name="create_diffs",
    )
    workflow.connect([
        (inputnode, create_diffs, [
            ("magnitude", "mag_file"),
            ("phase", "phase_file"),
        ]),
    ])  # fmt:skip

    # Unwrap phase data with ROMEO
    unwrap_phase = pe.MapNode(
        ROMEO(
            echo_times=echo_times,
            weights="romeo",
            correct_global=True,
            maxseeds=1,
            merge_regions=False,
            correct_regions=False,
        ),
        name="unwrap_phase",
        iterfield=["phase_file", "magnitude_file", "mask_file"],
    )
    workflow.connect([
        (inputnode, unwrap_phase, [("magnitude", "magnitude_file")]),
        (mask_buffer, unwrap_phase, [("mask_file", "mask_file")]),
        (create_diffs, unwrap_phase, [("phase_modified", "phase_file")]),
    ])  # fmt:skip

    # Global mode correction

    # Re-split the unwrapped phase data

    return workflow


def init_mcpc_3d_s_wf(wrap_limit, name):
    """Estimate and remove phase offset with MCPC-3D-S algorithm.

    Parameters
    ----------
    wrap_limit : bool
        If True, this turns off some heuristics for phase unwrapping.
    name : str
        The name of the workflow.

    Inputs
    ------
    magnitude : str
        The path to the magnitude image. A single volume, concatenated across echoes.
    phase : str
        The path to the phase image. A single volume, concatenated across echoes.
    echo_times : list of float
        The echo times of the multi-echo data.
    mask : str
        The path to the brain mask mask.

    Outputs
    -------
    offset : str
        The path to the estimated phase offset.
    unwrapped_diff : str
        The path to the unwrapped phase difference image.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "magnitude",
                "phase",
                "echo_times",
                "mask",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["offset", "unwrapped_diff"]),
        name="outputnode",
    )

    # Calculate magnitude and phase differences from first two echoes
    calc_diffs = pe.Node(
        niu.Function(
            input_names=["magnitude", "phase"],
            output_names=["mag_diff_file", "phase_diff_file"],
            function=calculate_diffs2,
        ),
        name="calc_diffs",
    )
    workflow.connect([
        (inputnode, calc_diffs, [
            ("magnitude", "magnitude"),
            ("phase", "phase"),
        ]),
    ])  # fmt:skip

    # Unwrap difference images
    unwrap_diffs = pe.Node(
        ROMEO(
            weights="romeo",
            correct_global=True,
        ),
        name="unwrap_diffs",
    )
    workflow.connect([
        (inputnode, unwrap_diffs, [("mask", "mask_file")]),
        (calc_diffs, unwrap_diffs, [
            ("mag_diff_file", "magnitude_file"),
            ("phase_diff_file", "phase_file"),
        ]),
    ])  # fmt:skip

    # Calculate voxel mask
    create_mask = pe.Node(
        niu.Function(
            input_names=["magnitude", "extra_dilation"],
            output_names=["mask"],
            function=create_brain_mask,
        ),
        name="create_mask",
    )
    create_mask.inputs.extra_dilation = -2  # hardcoded in warpkit
    workflow.connect([(inputnode, create_mask, [("magnitude", "magnitude")])])

    # Calculate initial offset estimate
    calc_offset = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["offset"],
            function=calculate_offset,
        ),
        name="calc_offset",
    )
    workflow.connect([
        (inputnode, calc_offset, [
            ("phase", "phase"),
            ("echo_times", "echo_times"),
        ]),
        (unwrap_diffs, calc_offset, [("out_file", "unwrapped_diff")]),
    ])  # fmt:skip

    # Get the new phase
    calc_proposed_phase = pe.Node(
        niu.Function(
            input_names=["phase", "offset"],
            output_names=["proposed_phase"],
            function=subtract_offset,
        ),
        name="calc_proposed_phase",
    )
    workflow.connect([
        (inputnode, calc_proposed_phase, [("phase", "phase")]),
        (calc_offset, calc_proposed_phase, [("offset", "offset")]),
    ])  # fmt:skip

    # Compute the dual-echo field map
    dual_echo_wf = init_dual_echo_wf(name="dual_echo_wf")
    workflow.connect([
        (inputnode, dual_echo_wf, [
            ("mask", "inputnode.mask_file"),
            ("echo_times", "inputnode.echo_times"),
            ("magnitude", "inputnode.magnitude"),
        ]),
        (calc_proposed_phase, dual_echo_wf, [("proposed_phase", "inputnode.phase")]),
    ])  # fmt:skip

    # Calculate a modified field map with 2pi added to the unwrapped difference image
    add_2pi = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["phase"],
            function=modify_unwrapped_diff,
        ),
        name="add_2pi",
    )
    workflow.connect([
        (inputnode, add_2pi, [
            ("phase", "phase")
            ("echo_times", "echo_times"),
        ]),
        (unwrap_diffs, add_2pi, [("out_file", "unwrapped_diff")]),
    ])  # fmt:skip

    modified_dual_echo_wf = init_dual_echo_wf(name="modified_dual_echo_wf")
    workflow.connect([
        (inputnode, modified_dual_echo_wf, [
            ("mask", "inputnode.mask_file"),
            ("echo_times", "inputnode.echo_times"),
            ("magnitude", "inputnode.magnitude"),
        ]),
        (add_2pi, modified_dual_echo_wf, [("phase", "inputnode.phase")]),
    ])  # fmt:skip

    # Select the fieldmap
    select_unwrapped_diff = pe.Node(
        niu.Function(
            input_names=[
                "original_fieldmap",
                "original_unwrapped_phase",
                "original_offset",
                "modified_fieldmap",
                "modified_unwrapped_phase",
                "unwrapped_diff",
                "voxel_mask",
                "echo_times",
                "wrap_limit",
            ],
            output_names=["new_unwrapped_diff"],
            function=select_fieldmap,
        ),
        name="select_unwrapped_diff",
    )
    select_unwrapped_diff.inputs.wrap_limit = wrap_limit
    workflow.connect([
        (inputnode, select_unwrapped_diff, [("echo_times", "echo_times")]),
        (unwrap_diffs, select_unwrapped_diff, [("out_file", "unwrapped_diff")]),
        (create_mask, select_unwrapped_diff, [("mask", "voxel_mask")]),
        (calc_offset, select_unwrapped_diff, [("offset", "original_offset")]),
        (dual_echo_wf, select_unwrapped_diff, [
            ("outputnode.fieldmap", "original_fieldmap"),
            ("outputnode.unwrapped_phase", "original_unwrapped_phase"),
        ]),
        (modified_dual_echo_wf, select_unwrapped_diff, [
            ("outputnode.fieldmap", "modified_fieldmap"),
            ("outputnode.unwrapped_phase", "modified_unwrapped_phase"),
        ]),
        (select_unwrapped_diff, outputnode, [("new_unwrapped_diff", "unwrapped_diff")]),
    ])  # fmt:skip

    # Compute the updated phase offset
    calc_updated_offset = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["offset"],
            function=calculate_offset,
        ),
        name="calc_updated_offset",
    )
    workflow.connect([
        (inputnode, calc_updated_offset, [
            ("phase", "phase"),
            ("echo_times", "echo_times"),
        ]),
        (select_unwrapped_diff, calc_updated_offset, [("new_unwrapped_diff", "unwrapped_diff")]),
        (calc_updated_offset, outputnode, [("offset", "offset")]),
    ])  # fmt:skip

    return workflow


def calculate_diffs2(magnitude, phase):
    """Calculate the magnitude and phase differences between two complex-valued images.

    Parameters
    ----------
    magnitude : :obj:`str`
        The path to the magnitude image (concatenated across echoes).
    phase : :obj:`str`
        The path to the phase image (concatenated across echoes).

    Returns
    -------
    mag_diff_file : :obj:`str`
        The path to the magnitude difference image.
    phase_diff_file : :obj:`str`
        The path to the phase difference image.
    """
    import os

    import nibabel as nb
    import numpy as np

    mag_diff_file = os.path.abspath("magnitude_diff.nii.gz")
    phase_diff_file = os.path.abspath("phase_diff.nii.gz")

    magnitude_img = nb.load(magnitude)
    phase_img = nb.load(phase)
    magnitude_data = magnitude_img.get_fdata()
    phase_data = phase_img.get_fdata()

    signal_diff = (
        magnitude_data[..., 0]
        * magnitude_data[..., 1]
        * np.exp(1j * (phase_data[..., 1] - phase_data[..., 0]))
    )
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    mag_diff_img = nb.Nifti1Image(mag_diff, magnitude_img.affine, magnitude_img.header)
    phase_diff_img = nb.Nifti1Image(phase_diff, phase_img.affine, phase_img.header)
    mag_diff_img.to_filename(mag_diff_file)
    phase_diff_img.to_filename(phase_diff_file)

    return mag_diff_file, phase_diff_file


def create_brain_mask(magnitude, extra_dilation):
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    magnitude : npt.NDArray[np.float32]
        Magnitude data with the shortest echo time
    extra_dilation : int
        Number of extra dilations (or erosions if negative) to perform, by default 0

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    """
    import os
    from typing import cast

    import nibabel as nb
    import numpy as np
    import numpy.typing as npt
    from scipy.ndimage import (
        binary_dilation,
        binary_fill_holes,
        binary_erosion,
        generate_binary_structure,
    )
    from skimage.filters import threshold_otsu

    from sdcflows.utils.misc import get_largest_connected_component

    mag_img = nb.load(magnitude)
    mag_data = mag_img.get_fdata()
    mag_data = mag_data[..., 0]

    mask_file = os.path.abspath("mask.nii.gz")

    # create structuring element
    strel = generate_binary_structure(3, 2)

    # get the otsu threshold
    threshold = threshold_otsu(mag_data)
    mask_data = mag_data > threshold
    mask_data = cast(npt.NDArray[np.float32], binary_fill_holes(mask_data, strel))

    # erode mask
    mask_data = cast(
        npt.NDArray[np.bool_],
        binary_erosion(mask_data, structure=strel, iterations=2, border_value=1),
    )

    # get largest connected component
    mask_data = get_largest_connected_component(mask_data)

    # dilate the mask
    mask_data = binary_dilation(mask_data, structure=strel, iterations=2)

    # extra dilation to get areas on the edge of the brain
    if extra_dilation > 0:
        mask_data = binary_dilation(mask_data, structure=strel, iterations=extra_dilation)
    # if negative, erode instead
    if extra_dilation < 0:
        mask_data = binary_erosion(mask_data, structure=strel, iterations=abs(extra_dilation))

    # since we can't have a completely empty mask, set all zeros to ones
    # if the mask is all empty
    if np.all(np.isclose(mask_data, 0)):
        mask_data = np.ones(mask_data.shape)

    mask_data = mask_data.astype(np.bool_).astype(np.uint8)
    mask_img = nb.Nifti1Image(mask_data, mag_img.affine, mag_img.header)
    mask_img.to_filename(mask_file)
    return mask_file


def calculate_offset(phase, unwrapped_diff, echo_times):
    """Calculate the phase offset between two echoes.

    Parameters
    ----------
    phase : :obj:`str`
        The path to the phase image (concatenated across echoes).
    unwrapped_diff : :obj:`str`
        The path to the unwrapped phase difference image from the first two echoes.
    echo_times : :obj:`list` of :obj:`float`
        The echo times.

    Returns
    -------
    offset : :obj:`str`
        The path to the phase offset image.
    """
    import os

    import nibabel as nb
    import numpy as np

    offset_file = os.path.abspath("offset.nii.gz")

    phase_img = nb.load(phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    phase_data = phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()

    proposed_offset = np.angle(
        np.exp(
            1j
            * (
                phase_data[..., 0]
                - ((echo_times[0] * unwrapped_diff_data) / (echo_times[1] - echo_times[0]))
            )
        )
    )
    proposed_offset_img = nb.Nifti1Image(proposed_offset, phase_img.affine, phase_img.header)
    proposed_offset_img.to_filename(offset_file)
    return offset_file


def subtract_offset(phase, offset):
    import os

    import nibabel as nb
    import numpy as np

    updated_phase_file = os.path.abspath("updated_phase.nii.gz")

    phase_img = nb.load(phase)
    offset_img = nb.load(offset)
    phase_data = phase_img.get_fdata()
    offset_data = offset_img.get_fdata()
    updated_phase = phase_data - offset_data[..., np.newaxis]
    updated_phase_img = nb.Nifti1Image(updated_phase, phase_img.affine, phase_img.header)
    updated_phase_img.to_filename(updated_phase_file)
    return updated_phase_file


def init_dual_echo_wf(name="dual_echo_wf"):
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "magnitude",
                "phase",
                "echo_times",
                "mask",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap", "unwrapped_phase"]),
        name="outputnode",
    )

    # Unwrap the phase with ROMEO
    unwrap_phase = pe.Node(
        ROMEO(
            weights="romeo",
            correct_global=True,
            maxseeds=1,
            merge_regions=False,
            correct_regions=False,
        ),
        name="unwrap_phase",
    )
    workflow.connect([
        (inputnode, unwrap_phase, [
            ("magnitude", "magnitude_file"),
            ("phase", "phase_file"),
            ("mask", "mask_file"),
            ("echo_times", "echo_times"),
        ]),
        (unwrap_phase, outputnode, [("out_file", "unwrapped_phase")]),
    ])  # fmt:skip

    # Calculate the fieldmap
    calc_fieldmap = pe.Node(
        niu.Function(
            input_names=["unwrapped_phase", "echo_times"],
            output_names=["fieldmap"],
            function=calculate_fieldmap,
        ),
        name="calc_fieldmap",
    )
    workflow.connect([
        (unwrap_phase, calc_fieldmap, [("out_file", "unwrapped_phase")]),
        (inputnode, calc_fieldmap, [("echo_times", "echo_times")]),
        (calc_fieldmap, outputnode, [("fieldmap", "fieldmap")]),
    ])  # fmt:skip

    return workflow


def calculate_fieldmap(unwrapped_phase, echo_times):
    import os

    import nibabel as nb
    import numpy as np

    fieldmap_file = os.path.abspath("fieldmap.nii.gz")

    unwrapped_phase_img = nb.load(unwrapped_phase)
    unwrapped_phase_data = unwrapped_phase_img.get_fdata()

    phase_diff = unwrapped_phase_data[..., 1] - unwrapped_phase_data[..., 0]
    fieldmap = (1000 / (2 * np.pi)) * phase_diff / (echo_times[1] - echo_times[0])
    fieldmap_img = nb.Nifti1Image(
        fieldmap,
        unwrapped_phase_img.affine,
        unwrapped_phase_img.header,
    )
    fieldmap_img.to_filename(fieldmap_file)

    return fieldmap_file


def modify_unwrapped_diff(phase, unwrapped_diff, echo_times):
    import os

    import nibabel as nb
    import numpy as np

    updated_phase_file = os.path.abspath("updated_phase.nii.gz")

    phase_img = nb.load(phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    phase_data = phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()

    new_proposed_offset = np.angle(
        np.exp(
            1j
            * (
                phase_data[..., 0]
                - (
                    (echo_times[0] * (unwrapped_diff_data + 2 * np.pi))
                    / (echo_times[1] - echo_times[0])
                )
            )
        )
    )
    new_proposed_phase = phase_data - new_proposed_offset[..., np.newaxis]
    new_proposed_phase_img = nb.Nifti1Image(
        new_proposed_phase,
        phase_img.affine,
        phase_img.header,
    )
    new_proposed_phase_img.to_filename(updated_phase_file)

    return updated_phase_file


def select_fieldmap(
    original_fieldmap,
    original_unwrapped_phase,
    original_offset,
    modified_fieldmap,
    modified_unwrapped_phase,
    unwrapped_diff,
    voxel_mask,
    echo_times,
    wrap_limit,
):
    import os

    import nibabel as nb
    import numpy as np

    FMAP_PROPORTION_HEURISTIC = 0.25
    FMAP_AMBIGUIOUS_HEURISTIC = 0.5

    new_unwrapped_diff_file = os.path.abspath("new_unwrapped_diff.nii.gz")

    orig_fmap_img = nb.load(original_fieldmap)
    orig_unwrapped_phase_img = nb.load(original_unwrapped_phase)
    orig_offset_img = nb.load(original_offset)
    mod_fmap_img = nb.load(modified_fieldmap)
    mod_unwrapped_phase_img = nb.load(modified_unwrapped_phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    voxel_mask_img = nb.load(voxel_mask)

    orig_fmap_data = orig_fmap_img.get_fdata()
    orig_unwrapped_phase_data = orig_unwrapped_phase_img.get_fdata()
    orig_offset_data = orig_offset_img.get_fdata()
    mod_fmap_data = mod_fmap_img.get_fdata()
    mod_unwrapped_phase_data = mod_unwrapped_phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()
    voxel_mask_data = voxel_mask_img.get_fdata()

    masked_orig_fmap = orig_fmap_data[voxel_mask_data]
    if masked_orig_fmap.mean() < -10:
        # check if the proposed fieldmap is below -10
        unwrapped_diff_data += 2 * np.pi

    elif masked_orig_fmap.mean() < 0 and not wrap_limit:
        # check if the proposed fieldmap is between -10 and 0
        # look at proportion of voxels that are positive
        voxel_prop = np.count_nonzero(masked_orig_fmap > 0) / masked_orig_fmap.shape[0]

        # if the proportion of positive voxels is less than 0.25, then add 2pi
        if voxel_prop < FMAP_PROPORTION_HEURISTIC:
            unwrapped_diff_data += 2 * np.pi
        elif voxel_prop < FMAP_AMBIGUIOUS_HEURISTIC:
            # compute mean of phase offset
            mean_phase_offset = orig_offset_data[voxel_mask_data].mean()
            if mean_phase_offset < -1:
                phase_fits = np.concatenate(
                    (
                        np.zeros((*orig_unwrapped_phase_data.shape[:-1], 1)),
                        orig_unwrapped_phase_data,
                    ),
                    axis=-1,
                )
                _, residuals_1, _, _, _ = np.polyfit(
                    echo_times,
                    phase_fits[voxel_mask_data, :].T,
                    1,
                    full=True,
                )

                masked_mod_fmap = mod_fmap_data[voxel_mask_data]
                # fit linear model to the proposed phase
                new_phase_fits = np.concatenate(
                    (
                        np.zeros((*mod_unwrapped_phase_data.shape[:-1], 1)),
                        mod_unwrapped_phase_data,
                    ),
                    axis=-1,
                )
                _, residuals_2, _, _, _ = np.polyfit(
                    echo_times,
                    new_phase_fits[voxel_mask_data, :].T,
                    1,
                    full=True,
                )

                if (
                    np.isclose(residuals_1.mean(), residuals_2.mean(), atol=1e-3, rtol=1e-3)
                    and masked_mod_fmap.mean() > 0
                ):
                    unwrapped_diff_data += 2 * np.pi
                else:
                    unwrapped_diff_data -= 2 * np.pi

    new_unwrapped_diff_img = nb.Nifti1Image(
        unwrapped_diff_data,
        unwrapped_diff_img.affine,
        unwrapped_diff_img.header,
    )
    new_unwrapped_diff_img.to_filename(new_unwrapped_diff_file)
    return new_unwrapped_diff_file
