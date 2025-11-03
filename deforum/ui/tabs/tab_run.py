"""Run tab for Deforum UI.

Contains basic generation settings like sampler, scheduler, steps, resolution,
seed, and batch mode/resume options.
"""

import gradio as gr
from modules.ui_components import FormRow
from deforum.utils.system.logging.emoji import run as emoji_run
from deforum.utils.ui.builders import create_row, create_gr_elem


def get_tab_run(d, da):
    """Create the Run tab with basic generation settings.

    Args:
        d: DeforumArgs namespace
        da: DeforumAnimArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    with gr.TabItem(f"{emoji_run()} Run"):
        motion_preview_mode = create_row(d.motion_preview_mode)
        sampler, scheduler, steps = create_row(d, 'sampler', 'scheduler', 'steps')
        W, H = create_row(d, 'W', 'H')
        seed, batch_name = create_row(d, 'seed', 'batch_name')

        with FormRow():
            restore_faces = create_gr_elem(d.restore_faces)
            tiling = create_gr_elem(d.tiling)
            enable_ddim_eta_scheduling = create_gr_elem(da.enable_ddim_eta_scheduling)
            enable_ancestral_eta_scheduling = create_gr_elem(da.enable_ancestral_eta_scheduling)

        with gr.Row(variant='compact') as eta_sch_row:
            ddim_eta_schedule = create_gr_elem(da.ddim_eta_schedule)
            ancestral_eta_schedule = create_gr_elem(da.ancestral_eta_schedule)

        # Batch Mode and Resume accordion
        with gr.Accordion('Batch Mode, Resume and more', open=True):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row():
                    override_settings_with_file = gr.Checkbox(
                        label="Enable batch mode",
                        value=False,
                        interactive=True,
                        elem_id='override_settings',
                        info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)"
                    )
                    custom_settings_file = gr.File(
                        label="Setting files",
                        interactive=True,
                        file_count="multiple",
                        file_types=[".txt"],
                        elem_id="custom_setting_file",
                        visible=False
                    )

            with gr.Tab('Resume Animation', selected=True):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring'
                )

    # Return all local Gradio components for event binding
    local_vars = locals()
    result = {
        k: v for k, v in local_vars.items()
        if not k.startswith('_') and k not in ('d', 'da', 'gr', 'FormRow', 'emoji_run', 'create_row', 'create_gr_elem')
    }

    return result
