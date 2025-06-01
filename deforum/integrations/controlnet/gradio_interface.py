import gradio as gr


# print (cnet_1.get_modules())

# *** TODO: re-enable table printing! disabled only temp! 13-04-23 ***
# table = Table(title="ControlNet params",padding=0, box=box.ROUNDED)

# TODO: auto infer the names and the values for the table
# field_names = []
# field_names += ["module", "model", "weight", "inv", "guide_start", "guide_end", "guess", "resize", "rgb_bgr", "proc res", "thr a", "thr b"]
# for field_name in field_names:
# table.add_column(field_name, justify="center")

# cn_model_name = str(controlnet_args.cn_1_model)

# rows = []
# rows += [controlnet_args.cn_1_module, cn_model_name[len('control_'):] if 'control_' in cn_model_name else cn_model_name, controlnet_args.cn_1_weight, controlnet_args.cn_1_invert_image, controlnet_args.cn_1_guidance_start, controlnet_args.cn_1_guidance_end, controlnet_args.cn_1_guess_mode, controlnet_args.cn_1_resize_mode, controlnet_args.cn_1_rgbbgr_mode, controlnet_args.cn_1_processor_res, controlnet_args.cn_1_threshold_a, controlnet_args.cn_1_threshold_b]
# rows = [str(x) for x in rows]

# table.add_row(*rows)
# console.print(table)


def hide_ui_by_cn_status(cn_enabled):
    return gr.update(visible=cn_enabled)


def hide_file_textboxes(cn_loopback_enabled):
    return gr.update(visible=not cn_loopback_enabled)


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Don't pass the variant here to avoid a type error in Gradio...
        self.variant = "tool"  # ...but setting the variant directly here is fine.

    def get_block_name(self):
        return "button"
