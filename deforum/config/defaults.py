from modules import sd_schedulers
from modules import sd_samplers


# Dynamically calls all the samplers from forge, so if it updates so does this
def get_samplers_list():
    samplers = {}
    for sampler in sd_samplers.all_samplers:
        samplers[sampler.name.lower()] = sampler.name
    return samplers


def get_schedulers_list():
    return {scheduler.name: scheduler.label for scheduler in sd_schedulers.schedulers}


def get_keyframe_distribution_list():
    return {
        'off': 'Off',
        'keyframes_only': 'Keyframes Only',
        'additive': 'Additive',
        'redistributed': 'Redistributed',
    }


def get_camera_shake_list():
    # Defined in .rendering.data.shakify.shake_data
    return {
        'NONE': 'None',
        'INVESTIGATION': 'Investigation',
        'THE_CLOSEUP': 'The Closeup',
        'THE_WEDDING': 'The Wedding',
        'WALK_TO_THE_STORE': 'Walk to the Store',
        'HANDYCAM_RUN': 'HandyCam Run',
        'OUT_CAR_WINDOW': 'Out Car Window',
        'BIKE_ON_GRAVEL_2D': 'Bike On Gravel (2D)',
        'SPACESHIP_SHAKE_2D': 'Spaceship Shake (2D)',
        'THE_ZEEK_2D': 'The Zeek (2D)',
    }

def DeforumAnimPrompts():
    # Keyframes are synchronized to line up at 60 FPS with amen13 from https://archive.org/details/amen-breaks/:
    # Direct link: https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3
    return r"""{
        "0": "A sterile hallway, brightly lit with fluorescent lights and empty",
        "12": "A sterile hallway, illuminated and overlooking a construction site through large windows",
        "43": "A sterile hallway, glowing with a digital grid pattern on the walls",
        "74": "An empty parking lot, featuring concrete surfaces and harsh lighting",
        "85": "An empty parking lot, under bright, flickering LED lights",
        "106": "A high-tech facility, with lights flickering in a vast, open area",
        "119": "A cold, reflective surface, illuminated by harsh overhead lights",
        "126": "A sterile environment, with vibrant lights creating a technological ambiance",
        "147": "A sterile space, with cold surfaces reflecting bright lights",
        "158": "A high-tech area, illuminated by neon lights in a clinical setting",
        "178": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "210": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "241": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "262": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "272": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "293": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "314": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "324": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens"
    }"""  # WARNING: make sure to not add a trailing semicolon after the last prompt, or the run might break.

# Guided images defaults
def get_guided_imgs_default_json():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''

def get_hybrid_info_html():
    return """
        <p style="padding-bottom:0">
            <b style="text-shadow: blue -1px -1px;">Hybrid Video Compositing in 2D/3D Mode</b>
            <span style="color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;">
                by <a href="https://github.com/reallybigname">reallybigname</a>
            </span>
        </p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Composite video with previous frame init image in <b>2D or 3D animation_mode</b> <i>(not for Video Input mode)</i></li>
            <li>Uses your <b>Init</b> settings for <b>video_init_path, extract_nth_frame, overwrite_extracted_frames</b></li>
            <li>In Keyframes tab, you can also set <b>color_coherence</b> = '<b>Video Input</b>'</li>
            <li><b>color_coherence_video_every_N_frames</b> lets you only match every N frames</li>
            <li>Color coherence may be used with hybrid composite off, to just use video color.</li>
            <li>Hybrid motion may be used with hybrid composite off, to just use video motion.</li>
        </ul>
        Hybrid Video Schedules
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>The alpha schedule controls overall alpha for video mix, whether using a composite mask or not.</li>
            <li>The <b>hybrid_comp_mask_blend_alpha_schedule</b> only affects the 'Blend' <b>hybrid_comp_mask_type</b>.</li>
            <li>Mask contrast schedule is from 0-255. Normal is 1. Affects all masks.</li>
            <li>Autocontrast low/high cutoff schedules 0-100. Low 0 High 100 is full range. <br>(<i><b>hybrid_comp_mask_auto_contrast</b> must be enabled</i>)</li>
        </ul>
        <a style='color:SteelBlue;' target='_blank' href='https://github.com/Tok/sd-forge-deforum/wiki/Animation-Settings#hybrid-video-mode-for-2d3d-animations'>Click Here</a> for more info/ a Guide.
        """

def get_wan_video_info_html():
    return """
        <p style="padding-bottom:0">
            <b style="text-shadow: blue -1px -1px;">Wan 2.1 Video Generation</b>
            <span style="color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;">
                powered by <a href="https://github.com/Wan-Video/Wan2.1" target="_blank">Wan 2.1</a>
            </span>
        </p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li><b>Text-to-Video</b>: Generate video clips directly from text prompts using Wan 2.1</li>
            <li><b>Image-to-Video</b>: Continue video generation using the last frame of the previous clip as initialization</li>
            <li><b>Frame Continuity</b>: Seamless transitions between clips ensure smooth video flow</li>
            <li><b>Audio Synchronization</b>: Align video clips with audio timing for perfect synchronization</li>
            <li><b>Prompt Scheduling</b>: Use Deforum's prompt scheduling system with frame-based timing</li>
        </ul>
        
        <p><b>Setup Requirements:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Download and install <a href="https://github.com/Wan-Video/Wan2.1" target="_blank">Wan 2.1</a></li>
            <li>Set the correct model path in the Wan Model Path field</li>
            <li>Ensure you have sufficient GPU memory (recommended: 12GB+ VRAM)</li>
            <li>Configure your prompts using standard Deforum JSON format</li>
        </ul>
        
        <p><b>How It Works:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li><b>First Clip</b>: Generated using text-to-video from your first prompt</li>
            <li><b>Subsequent Clips</b>: Generated using image-to-video with the last frame as init image</li>
            <li><b>Timing</b>: Clip duration calculated from prompt frame positions and FPS settings</li>
            <li><b>Output</b>: All clips are stitched together using FFmpeg for final video</li>
        </ul>
        
        <p><b>Performance Tips:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Use shorter clip durations (2-4 seconds) for better memory efficiency</li>
            <li>Lower inference steps (20-30) for faster generation</li>
            <li>Start with 512x512 resolution for testing, scale up for final renders</li>
            <li>Enable frame overlap for smoother transitions between clips</li>
        </ul>
        
        <p><b>Limitations:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Traditional Deforum camera movements are disabled in Wan mode</li>
            <li>3D depth warping and optical flow are not compatible</li>
            <li>Generation time is significantly longer than traditional diffusion</li>
            <li>Requires high-end hardware for optimal performance</li>
        </ul>
        
        <a style='color:SteelBlue;' target='_blank' href='https://github.com/Wan-Video/Wan2.1'>Visit Wan 2.1 Repository</a> for more information and installation instructions.
        """

def get_composable_masks_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>To enable, check use_mask in the Init tab</li>
        <li>Supports boolean operations: (! - negation, & - and, | - or, ^ - xor, \\\\ - difference, () - nested operations)</li>
        <li>default variables: in \\{\\}, like \\{init_mask\\}, \\{video_mask\\}, \\{everywhere\\}</li>
        <li>masks from files: in [], like [mask1.png]</li>
        <li>description-based: <i>word masks</i> in &lt;&gt;, like &lt;apple&gt;, &lt;hair&gt</li>
        </ul>
        """
        
def get_parseq_info_html():
    return """
        <p>Use a <a style='color:SteelBlue;' target='_blank' href='https://sd-parseq.web.app/deforum'>Parseq</a> manifest for your animation (leave blank to ignore).</p>
        <p style="margin-top:1em; margin-bottom:1em;">
            Fields managed in your Parseq manifest override the values and schedules set in other parts of this UI. You can select which values to override by using the "Managed Fields" section in Parseq.
        </p>
        """
        
def get_prompts_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>Please always keep values in math functions above 0.</li>
        <li>There is *no* Batch mode like in vanilla deforum. Please Use the txt2img tab for that.</li>
        <li>For negative prompts, please write your positive prompt, then --neg ugly, text, assymetric, or any other negative tokens of your choice. OR:</li>
        <li>Use the negative_prompts field to automatically append all words as a negative prompt. *Don't* add --neg in the negative_prompts field!</li>
        <li>Prompts are stored in JSON format. If you've got an error, check it in a <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">JSON Validator</a></li>
        </ul>
        """
        
def get_guided_imgs_info_html():
    return """        
        <p>You can use this as a guided image tool or as a looper depending on your settings in the keyframe images field. 
        Set the keyframes and the images that you want to show up. 
        Note: the number of frames between each keyframe should be greater than the tweening frames.</p>

        <p>Prerequisites and Important Info:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li>This mode works ONLY with 2D/3D animation modes. Interpolation and Video Input modes aren't supported.</li>
            <li>Init tab's strength slider should be greater than 0. Recommended value (.65 - .80).</li>
            <li>'seed_behavior' will be forcibly set to 'schedule'.</li>
        </ul>
        
        <p>Looping recommendations:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li>seed_schedule should start and end on the same seed.<br />
            Example: seed_schedule could use 0:(5), 1:(-1), 219:(-1), 220:(5)</li>
            <li>The 1st and last keyframe images should match.</li>
            <li>Set your total number of keyframes to be 21 more than the last inserted keyframe image.<br />
            Example: Default args should use 221 as the total keyframes.</li>
            <li>Prompts are stored in JSON format. If you've got an error, check it in the validator, 
            <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">like here</a></li>
        </ul>
        
        <p>The Guided images mode exposes the following variables for the prompts and the schedules:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li><b>s</b> is the <i>initial</i> seed for the whole video generation.</li>
            <li><b>max_f</b> is the length of the video, in frames.<br />
            Example: seed_schedule could use 0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)</li>
            <li><b>t</b> is the current frame number.<br />
            Example: strength_schedule could use 0:(0.25 * cos((72 / 60 * 3.141 * (t + 0) / 30))**13 + 0.7) to make alternating changes each 30 frames</li>
        </ul>
        """
        
def get_main_info_html():
    return """
        <p><strong>Made by <a href="https://deforum.github.io">deforum.github.io</a>, fork for WebUI Forge maintained by <a href="https://github.com/Tok/sd-forge-deforum">Zirteq</a>.</strong></p>
        <p><a  style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum/wiki/FAQ-&-Troubleshooting">FOR HELP CLICK HERE</a></p>
        <ul style="list-style-type:circle; margin-left:1em">
        <li>The code for this fork: <a  style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum">here</a>.</li>
        <li>Join the <a style="color:SteelBlue" href="https://discord.gg/deforum">official Deforum Discord</a> to share your creations and suggestions.</li>
        <li>Original Deforum Wiki: <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki">here</a>.</li>
        <li>Anime-inclined great guide (by FizzleDorf) with lots of examples: <a style="color:SteelBlue" href="https://rentry.org/AnimAnon-Deforum">here</a>.</li>
        <li>For advanced keyframing with Math functions, see <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/Maths-in-Deforum">here</a>.</li>
        <li>Alternatively, use <a style="color:SteelBlue" href="https://sd-parseq.web.app/deforum">sd-parseq</a> as a UI to define your animation schedules (see the Parseq section in the Init tab).</li>
        <li><a style="color:SteelBlue" href="https://www.framesync.xyz/">framesync.xyz</a> is also a good option, it makes compact math formulae for Deforum keyframes by selecting various waveforms.</li>
        <li>The other site allows for making keyframes using <a style="color:SteelBlue" href="https://www.chigozie.co.uk/keyframe-string-generator/">interactive splines and Bezier curves</a> (select Disco output format).</li>
        <li>If you want to use Width/Height which are not multiples of 64, please change noise_type to 'Uniform', in Keyframes --> Noise.</li>
        </ul>
        <italic>If you liked this fork, please <a style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum">give it a star on GitHub</a>!</italic> 😊
        """
def get_frame_interpolation_info_html():
    return """
        Use <a href="https://github.com/megvii-research/ECCV2022-RIFE">RIFE</a> / <a href="https://film-net.github.io/">FILM</a> Frame Interpolation to smooth out, slow-mo (or both) any video.</p>
         <p style="margin-top:1em">
            Supported engines:
            <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                <li>RIFE v4.6 and FILM.</li>
            </ul>
        </p>
         <p style="margin-top:1em">
            Important notes:
            <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                <li>Frame Interpolation will *not* run if any of the following are enabled: 'Store frames in ram' / 'Skip video for run all'.</li>
                <li>Audio (if provided) will *not* be transferred to the interpolated video if Slow-Mo is enabled.</li>
                <li>'add_soundtrack' and 'soundtrack_path' aren't being honoured in "Interpolate an existing video" mode. Original vid audio will be used instead with the same slow-mo rules above.</li>
                <li>In "Interpolate existing pics" mode, FPS is determined *only* by output FPS slider. Audio will be added if requested even with slow-mo "enabled", as it does *nothing* in this mode.</li>
            </ul>
        </p>
        """
def get_frames_to_video_info_html():
    return """
        <p style="margin-top:0em">
        Important Notes:
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:0.25em">
            <li>Enter relative to webui folder or Full-Absolute path, and make sure it ends with something like this: '20230124234916_%09d.png', just replace 20230124234916 with your batch ID. The %09d is important, don't forget it!</li>
            <li>In the filename, '%09d' represents the 9 counting numbers, For '20230124234916_000000001.png', use '20230124234916_%09d.png'</li>
            <li>If non-deforum frames, use the correct number of counting digits. For files like 'bunnies-0000.jpg', you'd use 'bunnies-%04d.jpg'</li>
        </ul>
        """
def get_leres_info_html():
    return 'Note that LeReS has a Non-Commercial <a href="https://github.com/aim-uofa/AdelaiDepth/blob/main/LeReS/LICENSE" target="_blank">license</a>. Use it only for fun/personal use.'

def get_gradio_html(section_name):
    if section_name.lower() == 'hybrid_video':
        return get_hybrid_info_html()
    elif section_name.lower() == 'wan_video':
        return get_wan_video_info_html()
    elif section_name.lower() == 'composable_masks':
        return get_composable_masks_info_html()
    elif section_name.lower() == 'parseq':
        return get_parseq_info_html()
    elif section_name.lower() == 'prompts':
        return get_prompts_info_html()
    elif section_name.lower() == 'guided_imgs':
        return get_guided_imgs_info_html()
    elif section_name.lower() == 'main':
        return get_main_info_html()
    elif section_name.lower() == 'frame_interpolation':
        return get_frame_interpolation_info_html()
    elif section_name.lower() == 'frames_to_video':
        return get_frames_to_video_info_html()
    elif section_name.lower() == 'leres':
        return get_leres_info_html()
    else:
        return ""

mask_fill_choices = ['fill', 'original', 'latent noise', 'latent nothing']
        
