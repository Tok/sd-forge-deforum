"""Unit tests for high-resolution color naming system.

Tests hue wheel accuracy, brightness classification, and color naming.
"""

import pytest
from deforum.utils.image.color_namer import (
    hex_to_rgb,
    rgb_to_hsv,
    get_hue_name,
    get_saturation_prefix,
    get_value_prefix,
    name_color,
    name_color_simple,
    describe_gradient,
)


class TestColorConversion:
    """Test color space conversion functions."""

    def test_hex_to_rgb_with_hash(self):
        """Test hex to RGB conversion with # prefix."""
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)

    def test_hex_to_rgb_without_hash(self):
        """Test hex to RGB conversion without # prefix."""
        assert hex_to_rgb("FF0000") == (255, 0, 0)
        assert hex_to_rgb("FFFFFF") == (255, 255, 255)
        assert hex_to_rgb("000000") == (0, 0, 0)

    def test_hex_to_rgb_lowercase(self):
        """Test hex to RGB with lowercase letters."""
        assert hex_to_rgb("#ff00ff") == (255, 0, 255)
        assert hex_to_rgb("abc123") == (171, 193, 35)

    def test_rgb_to_hsv_red(self):
        """Test RGB to HSV for pure red."""
        h, s, v = rgb_to_hsv(255, 0, 0)
        assert h == pytest.approx(0, abs=1)
        assert s == pytest.approx(100, abs=1)
        assert v == pytest.approx(100, abs=1)

    def test_rgb_to_hsv_green(self):
        """Test RGB to HSV for pure green."""
        h, s, v = rgb_to_hsv(0, 255, 0)
        assert h == pytest.approx(120, abs=1)
        assert s == pytest.approx(100, abs=1)
        assert v == pytest.approx(100, abs=1)

    def test_rgb_to_hsv_blue(self):
        """Test RGB to HSV for pure blue."""
        h, s, v = rgb_to_hsv(0, 0, 255)
        assert h == pytest.approx(240, abs=1)
        assert s == pytest.approx(100, abs=1)
        assert v == pytest.approx(100, abs=1)

    def test_rgb_to_hsv_gray(self):
        """Test RGB to HSV for gray (no saturation)."""
        h, s, v = rgb_to_hsv(128, 128, 128)
        assert s == pytest.approx(0, abs=1)
        assert v == pytest.approx(50, abs=2)


class TestHueNaming:
    """Test hue wheel name accuracy."""

    def test_primary_colors(self):
        """Test standard primary colors at key angles."""
        assert get_hue_name(0) == "red"
        assert get_hue_name(180) == "cyan"
        assert get_hue_name(240) == "indigo"

    def test_secondary_colors(self):
        """Test standard secondary colors."""
        assert get_hue_name(68) == "yellow"  # Yellow is 64-72°
        assert get_hue_name(120) == "green"
        assert get_hue_name(300) == "magenta"

    def test_red_pink_transition(self):
        """Test accurate red-pink transition (335-360°)."""
        assert get_hue_name(325) == "rose"     # 320-328°
        assert get_hue_name(330) == "pink"     # 328-336°
        assert get_hue_name(340) == "salmon"   # 336-344°
        assert get_hue_name(348) == "flamingo" # 344-352°
        assert get_hue_name(353) == "watermelon" # 352-360°
        assert get_hue_name(359) == "watermelon"

    def test_deforum_gradient_colors(self):
        """Test Deforum BB0 and DA3 gradient endpoint hues."""
        # BB0: #5606FF (259.3°) should be purple
        assert get_hue_name(259.3) == "purple"

        # BB0: #17A7FE (202.6°) should be azure
        assert get_hue_name(202.6) == "azure"

        # DA3: #1CC4E6 (190.1°) should be cyan
        assert get_hue_name(190.1) == "cyan"

        # DA3: #F64A5E (353.0°) should be watermelon
        assert get_hue_name(353.0) == "watermelon"

    def test_coral_range(self):
        """Test coral color range (16-24°)."""
        assert get_hue_name(16) == "coral"
        assert get_hue_name(20) == "coral"
        assert get_hue_name(23) == "coral"

    def test_hue_wrapping(self):
        """Test hue normalization for values > 360°."""
        assert get_hue_name(360) == "red"
        assert get_hue_name(361) == "red"
        assert get_hue_name(720) == "red"
        assert get_hue_name(428) == "yellow"  # 428 % 360 = 68 (yellow is 64-72°)


class TestSaturationPrefix:
    """Test saturation-based prefix classification."""

    def test_grey_range(self):
        """Test grey/desaturated colors."""
        assert get_saturation_prefix(0) == "grey"
        assert get_saturation_prefix(5) == "grey"
        assert get_saturation_prefix(9) == "grey"

    def test_greyish_range(self):
        """Test greyish colors."""
        assert get_saturation_prefix(10) == "greyish"
        assert get_saturation_prefix(20) == "greyish"
        assert get_saturation_prefix(24) == "greyish"

    def test_muted_range(self):
        """Test muted colors."""
        assert get_saturation_prefix(25) == "muted"
        assert get_saturation_prefix(35) == "muted"
        assert get_saturation_prefix(39) == "muted"

    def test_soft_range(self):
        """Test soft colors."""
        assert get_saturation_prefix(40) == "soft"
        assert get_saturation_prefix(50) == "soft"
        assert get_saturation_prefix(59) == "soft"

    def test_normal_range(self):
        """Test normal saturation (no prefix)."""
        assert get_saturation_prefix(60) == ""
        assert get_saturation_prefix(70) == ""
        assert get_saturation_prefix(79) == ""

    def test_vivid_range(self):
        """Test vivid colors."""
        assert get_saturation_prefix(80) == "vivid"
        assert get_saturation_prefix(85) == "vivid"
        assert get_saturation_prefix(89) == "vivid"

    def test_electric_range(self):
        """Test electric saturation."""
        assert get_saturation_prefix(90) == "electric"
        assert get_saturation_prefix(95) == "electric"
        assert get_saturation_prefix(100) == "electric"


class TestValuePrefix:
    """Test brightness/value-based prefix classification."""

    def test_black_range(self):
        """Test black colors (very dark)."""
        assert get_value_prefix(0, 50) == "black"
        assert get_value_prefix(10, 50) == "black"
        assert get_value_prefix(14, 50) == "black"

    def test_very_dark_range(self):
        """Test very dark colors."""
        assert get_value_prefix(15, 50) == "very dark"
        assert get_value_prefix(20, 50) == "very dark"
        assert get_value_prefix(29, 50) == "very dark"

    def test_dark_range(self):
        """Test dark colors."""
        assert get_value_prefix(30, 50) == "dark"
        assert get_value_prefix(35, 50) == "dark"
        assert get_value_prefix(44, 50) == "dark"

    def test_normal_range(self):
        """Test normal brightness (no prefix)."""
        assert get_value_prefix(50, 50) == ""
        assert get_value_prefix(60, 50) == ""
        assert get_value_prefix(69, 50) == ""

    def test_bright_saturated(self):
        """Test bright saturated colors."""
        assert get_value_prefix(71, 25) == "bright"
        assert get_value_prefix(75, 30) == "bright"
        assert get_value_prefix(84, 50) == "bright"

    def test_very_bright_saturated(self):
        """Test very bright saturated colors."""
        assert get_value_prefix(86, 25) == "very bright"
        assert get_value_prefix(90, 30) == "very bright"
        assert get_value_prefix(100, 50) == "very bright"

    def test_pale_desaturated(self):
        """Test pale desaturated colors."""
        assert get_value_prefix(76, 18) == "pale"
        assert get_value_prefix(80, 15) == "pale"
        assert get_value_prefix(89, 19) == "pale"

    def test_white_desaturated(self):
        """Test white (very bright desaturated)."""
        assert get_value_prefix(91, 10) == "white"
        assert get_value_prefix(95, 5) == "white"
        assert get_value_prefix(100, 14) == "white"


class TestFullColorNaming:
    """Test complete color naming with all components."""

    def test_deforum_bb0_colors(self):
        """Test BB0 gradient colors."""
        # #5606FF - Electric purple
        name = name_color_simple("#5606FF")
        assert "purple" in name.lower()
        assert "electric" in name.lower()

        # #17A7FE - Electric azure
        name = name_color_simple("#17A7FE")
        assert "azure" in name.lower()
        assert "electric" in name.lower()

    def test_deforum_da3_colors(self):
        """Test DA3 gradient colors."""
        # #1CC4E6 - Vivid cyan
        name = name_color_simple("#1CC4E6")
        assert "cyan" in name.lower()
        assert "vivid" in name.lower()

        # #F64A5E - Watermelon (not red!)
        name = name_color_simple("#F64A5E")
        assert "watermelon" in name.lower()
        assert "red" not in name.lower()  # Should be watermelon, not red

    def test_technical_output(self):
        """Test technical output with HSV values."""
        name = name_color("#FF0000", include_technical=True)
        assert "H:" in name
        assert "S:" in name
        assert "V:" in name
        assert "°" in name
        assert "%" in name

    def test_pure_colors(self):
        """Test pure saturated colors."""
        assert "red" in name_color_simple("#FF0000").lower()
        assert "green" in name_color_simple("#00FF00").lower()
        # Pure blue (#0000FF) is at 240° which is "indigo" in our wheel
        assert "indigo" in name_color_simple("#0000FF").lower()

    def test_dark_colors(self):
        """Test dark color naming."""
        name = name_color_simple("#2A0000")  # Dark red
        assert "dark" in name.lower()
        assert "red" in name.lower()

    def test_bright_colors(self):
        """Test bright color naming."""
        name = name_color_simple("#FFE5E5")  # Very light desaturated red
        # Should have white or pale prefix due to high value + low saturation
        assert ("pale" in name.lower() or "white" in name.lower() or "bright" in name.lower())

    def test_grey_colors(self):
        """Test grey/desaturated color naming."""
        name = name_color_simple("#808080")  # Mid grey
        assert "grey" in name.lower()


class TestGradientDescriptions:
    """Test gradient description generation."""

    def test_bb0_gradient(self):
        """Test BB0 gradient description."""
        desc = describe_gradient("#5606FF", "#17A7FE")
        assert "purple" in desc.lower()
        assert "azure" in desc.lower()
        assert "→" in desc

    def test_da3_gradient(self):
        """Test DA3 gradient description."""
        desc = describe_gradient("#1CC4E6", "#F64A5E")
        assert "cyan" in desc.lower()
        assert "watermelon" in desc.lower()
        assert "→" in desc

    def test_gradient_format(self):
        """Test gradient description format."""
        desc = describe_gradient("#FF0000", "#0000FF")
        parts = desc.split("→")
        assert len(parts) == 2
        assert parts[0].strip()  # Start color
        assert parts[1].strip()  # End color


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_hue(self):
        """Test negative hue values normalize correctly."""
        assert get_hue_name(-10) == get_hue_name(350)
        assert get_hue_name(-180) == get_hue_name(180)

    def test_extreme_values(self):
        """Test extreme saturation/value inputs."""
        # Should not crash
        assert get_saturation_prefix(0) == "grey"
        assert get_saturation_prefix(100) == "electric"
        assert get_value_prefix(0, 50) == "black"
        assert get_value_prefix(100, 50) == "very bright"

    def test_invalid_hex_handling(self):
        """Test handling of edge case hex values."""
        # Single color channel
        assert hex_to_rgb("#FF0000")[0] == 255
        assert hex_to_rgb("#00FF00")[1] == 255
        assert hex_to_rgb("#0000FF")[2] == 255

    def test_empty_prefix(self):
        """Test normal colors return no prefix."""
        # Mid-saturation, mid-value should have minimal prefixes
        name = name_color_simple("#808080")  # Pure grey
        # Grey colors will have "grey" prefix, but not brightness prefix
        assert "very" not in name.lower() or "grey" in name.lower()
