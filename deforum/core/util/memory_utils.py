# noinspection PyUnresolvedReferences
from modules import lowvram, devices, sd_hijack
# noinspection PyUnresolvedReferences
from modules.shared import cmd_opts, sd_model


def is_low_or_med_vram():
    return cmd_opts.lowvram or cmd_opts.medvram  # cmd_opts are imported from elsewhere. keep readonly


def handle_med_or_low_vram_before_step(data):
    if data.is_3d_with_med_or_low_vram():
        # Unload the main checkpoint and load the depth model
        lowvram.send_everything_to_cpu()
        sd_hijack.model_hijack.undo_hijack(sd_model)
        devices.torch_gc()
        if data.animation_mode.is_predicting_depths:
            data.depth_model.to(data.args.root.device)


def handle_vram_if_depth_is_predicted(data):
    if data.animation_mode.is_predicting_depths:
        if data.is_3d_with_med_or_low_vram():
            data.depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)


def handle_vram_before_depth_map_generation(data):
    if is_low_or_med_vram():
        lowvram.send_everything_to_cpu()
        sd_hijack.model_hijack.undo_hijack(sd_model)
        devices.torch_gc()
        data.depth_model.to(data.args.root.device)


def handle_vram_after_depth_map_generation(data):
    if is_low_or_med_vram():
        data.depth_model.to('cpu')
        devices.torch_gc()
        lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
        sd_hijack.model_hijack.hijack(sd_model)


def select_depth_device(root):
    return 'cpu' if is_low_or_med_vram() else root.device
