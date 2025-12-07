"""Generate accurate color names for all Deforum gradient shades.

Analyzes all gradient colors and prints accurate names for documentation.
"""

import sys
sys.path.insert(0, '/home/zirteq/workspace/forge-neo/extensions/sd-forge-deforum')

from deforum.utils.image.color_namer import name_color, describe_gradient


# BB0 Gradient Colors
print("="*80)
print("BB0 GRADIENT - BLANK BANSHEE 0")
print("="*80)
print()

bb0_endpoints = {
    '#5606FF': 'Electric purple (album top)',
    '#17A7FE': 'Azure cyan (album bottom)'
}

print("Endpoints:")
for hex_color, description in bb0_endpoints.items():
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {description}")
    print(f"    Auto: {auto_name}")
print()

bb0_7_shade = [
    ('#5606FF', 'Electric purple (album top)'),
    ('#4C21FF', 'Purple-blue'),
    ('#413CFF', 'Blue-purple (banner start)'),
    ('#3757FF', 'Mid blue'),
    ('#2C71FE', 'Blue'),
    ('#228CFE', 'Bright blue'),
    ('#17A7FE', 'Azure cyan (album bottom, banner end)')
]

print("7-Shade Gradient (CLI banner, charts):")
for hex_color, current_name in bb0_7_shade:
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {current_name}")
    print(f"    Auto: {auto_name}")
print()

bb0_5_tqdm = [
    ('#5606FF', 'Electric purple [Current Tweens - FASTEST]'),
    ('#462EFF', 'Purple-blue [Current Steps - FAST]'),
    ('#3757FF', 'Mid blue [Total Steps - MEDIUM]'),
    ('#277FFE', 'Bright blue [Total Diffusion Frames - SLOW]'),
    ('#17A7FE', 'Azure cyan [Total Frames - SLOWEST]')
]

print("5-Shade TQDM Gradient:")
for hex_color, current_name in bb0_5_tqdm:
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {current_name}")
    print(f"    Auto: {auto_name}")
print()

# DA3 Gradient Colors
print("="*80)
print("DA3 GRADIENT - DEPTH ANYTHING V3")
print("="*80)
print()

da3_endpoints = {
    '#1CC4E6': 'Electric cyan (gradient start)',
    '#F64A5E': 'Coral red (gradient end)'
}

print("Endpoints:")
for hex_color, description in da3_endpoints.items():
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {description}")
    print(f"    Auto: {auto_name}")
print()

da3_7_shade = [
    ('#1CC4E6', 'Electric cyan (gradient start)'),
    ('#40AFCF', 'Cyan-blue blend'),
    ('#649BB8', 'Blue-teal'),
    ('#8987A2', 'Mid purple-grey'),
    ('#AD728B', 'Purple-pink'),
    ('#D15E74', 'Rose pink'),
    ('#F64A5E', 'Coral red (gradient end)')
]

print("7-Shade Gradient (CLI banner, charts):")
for hex_color, current_name in da3_7_shade:
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {current_name}")
    print(f"    Auto: {auto_name}")
print()

da3_5_tqdm = [
    ('#1CC4E6', 'Electric cyan [Current Tweens - FASTEST]'),
    ('#52A5C4', 'Blue-teal [Current Steps - FAST]'),
    ('#8987A2', 'Mid purple-grey [Total Steps - MEDIUM]'),
    ('#BF6880', 'Pink-purple [Total Diffusion Frames - SLOW]'),
    ('#F64A5E', 'Coral red [Total Frames - SLOWEST]')
]

print("5-Shade TQDM Gradient:")
for hex_color, current_name in da3_5_tqdm:
    auto_name = name_color(hex_color, include_technical=True)
    print(f"  {hex_color}: {current_name}")
    print(f"    Auto: {auto_name}")
print()

# Summary
print("="*80)
print("GRADIENT DESCRIPTIONS")
print("="*80)
print()
print(f"BB0: {describe_gradient('#5606FF', '#17A7FE')}")
print(f"DA3: {describe_gradient('#1CC4E6', '#F64A5E')}")
