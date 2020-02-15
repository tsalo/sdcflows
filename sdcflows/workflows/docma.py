import numpy as np

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from .unwarp import init_sdc_unwarp_wf
from .phdiff import init_phdiff_wf


def group_files(file1, file2, file1_metadata=None, file2_metadata=None):
    """
    Group arguments into a list of tuples.
    """
    if file1_metadata is None and file2_metadata is None:
        return (file1, file2)
    else:
        return ((file1, file1_metadata), (file2, file2_metadata))


def split_files(file_4d, volume):
    """
    Split an input 4D file and write out the set of 3D files.
    """
    import os.path as op
    from nilearn.image import index_img
    from nipype.utils.filemanip import split_filename

    _, base, _ = split_filename(file_4d)
    file_3d = op.abspath(base + "_{0:05d}.nii.gz".format(volume))
    img = index_img(file_4d, volume)
    img.to_filename(file_3d)
    return file_3d


def join_results(files_3d):
    """
    Concatenate a set of input 3D files and write out the concatenated 4D file.
    """
    import os.path as op
    from nilearn.image import concat_imgs
    from nipype.utils.filemanip import split_filename

    _, base, _ = split_filename(files_3d[0])
    file_4d = op.abspath(base + ".nii.gz")
    img = concat_imgs(files_3d)
    img.to_filename(file_4d)
    return file_4d


def init_docma_wf(omp_nthreads, num_trs, name='docma_wf'):
    """
    Use the DOCMA approach to generate 4D field maps from complex, multi-echo
    EPI data.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.phdiff import init_docma_wf
            wf = init_docma_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use
    num_trs : int
        Number of volumes in the magnitude and phase time series

    Inputs
    ------
    magnitude1 : pathlike
        Path to the corresponding magnitude file for the shorter echo time.
    magnitude2 : pathlike
        Path to the corresponding magnitude file for the longer echo time.
    phase1 : pathlike
        Path to the corresponding phase file with the shorter echo time.
    phase2 : pathlike
        Path to the corresponding phase file with the longer echo time.
    phase1_metadata : dict
        Metadata dictionary corresponding to the short echo time phase input
    phase2_metadata : dict
        Metadata dictionary corresponding to the short echo time phase input
    mask_file : pathlike, optional
        Optional mask file

    Outputs
    -------
    fmap_ref : pathlike
        The magnitude time series, skull-stripped
    fmap_mask : pathlike
        The volume-wise brain masks applied to the fieldmaps
    fmap : pathlike
        The estimated fieldmap time series in Hz
    reference : str
        the ``in_reference`` after unwarping
    reference_brain : str
        the ``in_reference`` after unwarping and skullstripping
    warp : str
        the ``in_warp`` field is forwarded for compatibility
    mask : str
        mask of the unwarped input file

    References
    ----------
    .. [Visser2012] Visser, E., Poser, B. A., Barth, M., & Zwiers, M. P. (2012).
       Reference‐free unwarping of EPI data using dynamic off‐resonance correction
       with multiecho acquisition (DOCMA). Magnetic resonance in medicine, 68(4),
       1247-1254. doi:`10.1002/mrm.24119 <10.1002/mrm.24119>`__.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Volume-wise deformation fields to correct for motion-related susceptibility
distortions were estimated based on multi-echo EPI magnitude and phase data,
using a custom workflow of *SDCFlows* derived from the Dynamic Off-Resonance
Correction With Multiecho Acquisition [@docma] method.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['phase1', 'phase2', 'phase1_metadata', 'phase2_metadata',
                    'magnitude1', 'magnitude2', 'mask_file']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask',
                'reference', 'reference_brain', 'warp', 'mask']),
        name='outputnode')

    # Split 4D files into 3D (Python 3.5+)
    buffernode = pe.Node(niu.IdentityInterface(
        fields=['magnitude1', 'magnitude2', 'phase1', 'phase2', 'volume']),
        iterables=[('volume', np.arange(num_trs, dtype=int))],
        name='buffernode')
    split_mag1 = pe.Node(
        interface=niu.Function(['file_4d', 'volume'], ['file_3d'], split_files),
        name='split_mag1')
    split_mag2 = pe.Node(
        interface=niu.Function(['file_4d', 'volume'], ['file_3d'], split_files),
        name='split_mag2')
    split_phase1 = pe.Node(
        interface=niu.Function(['file_4d', 'volume'], ['file_3d'], split_files),
        name='split_phase1')
    split_phase2 = pe.Node(
        interface=niu.Function(['file_4d', 'volume'], ['file_3d'], split_files),
        name='split_phase2')

    # Collect split files into tuples
    group_magnitudes = pe.Node(
        interface=niu.Function(['file1', 'file2', 'file1_metadata', 'file2_metadata'],
                               ['out_tuples'], group_files),
        name='group_magnitudes')
    group_phases = pe.Node(
        interface=niu.Function(['file1', 'file2', 'file1_metadata', 'file2_metadata'],
                               ['out_tuples'], group_files),
        name='group_phases')

    # Generate a phase-difference-based field map from each set of 3D files
    phdiff_wf = init_phdiff_wf(omp_nthreads, name='phdiff_wf')
    # No clue if this is right but we want to disable the description
    phdiff_wf.__desc__ = ''

    # Generate warps in this workflow instead of a separate one to avoid having
    # to resplit and rejoin everything
    sdc_unwarp_wf = init_sdc_unwarp_wf(omp_nthreads, debug=False, name='sdc_unwarp_wf')
    sdc_unwarp_wf.__desc__ = ''

    # Concatenate 3D results to 4D files
    join_fmap = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_fmap',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_fmap_ref = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_fmap_ref',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_fmap_mask = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_fmap_mask',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_reference = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_reference',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_reference_brain = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_reference_brain',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_warp = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_warp',
        joinfield=['files_3d'],
        joinsource='buffernode')
    join_mask = pe.JoinNode(
        interface=niu.Function(['files_3d'], ['file_4d'], join_results),
        name='join_mask',
        joinfield=['files_3d'],
        joinsource='buffernode')

    workflow.connect([
        (inputnode, buffernode, [('magnitude1', 'magnitude1'),
                                 ('magnitude2', 'magnitude2'),
                                 ('phase1', 'phase1'),
                                 ('phase2', 'phase2')]),
        (buffernode, split_mag1, [('magnitude1', 'file_4d'),
                                  ('volume', 'volume')]),
        (buffernode, split_mag2, [('magnitude2', 'file_4d'),
                                  ('volume', 'volume')]),
        (buffernode, split_phase1, [('phase1', 'file_4d'),
                                    ('volume', 'volume')]),
        (buffernode, split_phase2, [('phase2', 'file_4d'),
                                    ('volume', 'volume')]),
        (split_mag1, group_magnitudes, [('file_3d', 'file1')]),
        (split_mag2, group_magnitudes, [('file_3d', 'file2')]),
        (group_magnitudes, phdiff_wf, [('out_tuples', 'inputnode.magnitude')]),
        (split_phase1, group_phases, [('file_3d', 'file1')]),
        (split_phase2, group_phases, [('file_3d', 'file2')]),
        (inputnode, group_phases, [('phase1_metadata', 'file1_metadata'),
                                   ('phase2_metadata', 'file2_metadata')]),
        (group_phases, phdiff_wf, [('out_tuples', 'inputnode.phasediff')]),
        (phdiff_wf, sdc_unwarp_wf, [('outputnode.fmap', 'inputnode.in_warp'),
                                    ('outputnode.fmap_ref', 'inputnode.in_reference'),
                                    ('outputnode.fmap_mask', 'inputnode.in_reference_mask')]),
        (phdiff_wf, join_fmap, [('outputnode.fmap', 'files_3d')]),
        (phdiff_wf, join_fmap_mask, [('outputnode.fmap_mask', 'files_3d')]),
        (phdiff_wf, join_fmap_ref, [('outputnode.fmap_ref', 'files_3d')]),
        (sdc_unwarp_wf, join_reference, [('outputnode.out_reference', 'files_3d')]),
        (sdc_unwarp_wf, join_reference_brain, [('outputnode.out_reference_brain', 'files_3d')]),
        (sdc_unwarp_wf, join_warp, [('outputnode.out_warp', 'files_3d')]),
        (sdc_unwarp_wf, join_mask, [('outputnode.out_mask', 'files_3d')]),
        (join_fmap, outputnode, [('file_4d', 'fmap')]),
        (join_fmap_mask, outputnode, [('file_4d', 'fmap_mask')]),
        (join_fmap_ref, outputnode, [('file_4d', 'fmap_ref')]),
        (join_reference, outputnode, [('file_4d', 'reference')]),
        (join_reference_brain, outputnode, [('file_4d', 'reference_brain')]),
        (join_warp, outputnode, [('file_4d', 'warp')]),
        (join_mask, outputnode, [('file_4d', 'mask')]),
    ])
    return workflow
