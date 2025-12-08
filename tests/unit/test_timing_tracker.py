"""Unit tests for TimingTracker performance profiling system."""

import time
import pytest
from deforum.utils.timing_tracker import TimingTracker, get_timing_tracker


class TestTimingTracker:
    """Test suite for TimingTracker singleton."""

    def test_singleton_pattern(self):
        """Verify TimingTracker is a singleton."""
        tracker1 = TimingTracker()
        tracker2 = TimingTracker()
        tracker3 = get_timing_tracker()

        assert tracker1 is tracker2
        assert tracker2 is tracker3
        assert id(tracker1) == id(tracker2) == id(tracker3)

    def test_initialization(self):
        """Verify tracker initializes with correct categories."""
        tracker = TimingTracker()
        tracker.reset()

        expected_categories = {
            'diffusion', 'depth_estimation', 'depth_warping',
            'wan_interpolation', 'preprocessing', 'postprocessing',
            'video_stitching', 'other'
        }

        assert set(tracker.categories.keys()) == expected_categories
        assert all(val == 0.0 for val in tracker.categories.values())
        assert tracker.total_time == 0.0
        assert tracker.start_time is None

    def test_reset(self):
        """Verify reset clears all timing data."""
        tracker = TimingTracker()

        # Set some values
        tracker.categories['diffusion'] = 10.5
        tracker.categories['depth_warping'] = 5.2
        tracker.total_time = 20.0
        tracker.start_time = time.perf_counter()

        # Reset
        tracker.reset()

        # Verify all cleared
        assert all(val == 0.0 for val in tracker.categories.values())
        assert tracker.total_time == 0.0
        assert tracker.start_time is None

    def test_start_generation(self):
        """Verify start_generation initializes timing."""
        tracker = TimingTracker()

        # Add some data first
        tracker.categories['diffusion'] = 5.0
        tracker.total_time = 10.0

        before = time.perf_counter()
        tracker.start_generation()
        after = time.perf_counter()

        # Should reset all categories
        assert all(val == 0.0 for val in tracker.categories.values())
        assert tracker.total_time == 0.0

        # Should set start_time to current time
        assert tracker.start_time is not None
        assert before <= tracker.start_time <= after

    def test_end_generation(self):
        """Verify end_generation calculates total time."""
        tracker = TimingTracker()
        tracker.start_generation()

        # Simulate work
        time.sleep(0.1)

        tracker.end_generation()

        # Total time should be calculated
        assert tracker.total_time > 0.0
        assert tracker.total_time >= 0.1  # At least sleep duration

    def test_track_context_manager(self):
        """Verify track() context manager accumulates time correctly."""
        tracker = TimingTracker()
        tracker.reset()

        # Track diffusion time
        with tracker.track('diffusion'):
            time.sleep(0.05)

        assert tracker.categories['diffusion'] > 0.0
        assert tracker.categories['diffusion'] >= 0.05
        assert tracker.categories['depth_warping'] == 0.0  # Untouched

    def test_track_multiple_calls(self):
        """Verify track() accumulates across multiple calls."""
        tracker = TimingTracker()
        tracker.reset()

        # Multiple diffusion calls
        with tracker.track('diffusion'):
            time.sleep(0.03)

        with tracker.track('diffusion'):
            time.sleep(0.02)

        # Should accumulate
        assert tracker.categories['diffusion'] >= 0.05

        # Other categories untouched
        assert tracker.categories['depth_warping'] == 0.0

    def test_track_unknown_category(self):
        """Verify unknown category falls back to 'other'."""
        tracker = TimingTracker()
        tracker.reset()

        with tracker.track('unknown_operation'):
            time.sleep(0.02)

        # Should accumulate in 'other'
        assert tracker.categories['other'] >= 0.02
        assert 'unknown_operation' not in tracker.categories

    def test_add_time(self):
        """Verify add_time() manually adds time to category."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('diffusion', 5.5)
        tracker.add_time('diffusion', 2.3)

        assert tracker.categories['diffusion'] == 7.8
        assert tracker.categories['depth_warping'] == 0.0

    def test_add_time_unknown_category(self):
        """Verify add_time() with unknown category uses 'other'."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('invalid_category', 3.0)

        assert tracker.categories['other'] == 3.0
        assert 'invalid_category' not in tracker.categories

    def test_print_report_no_data(self, capsys):
        """Verify print_report() handles no timing data gracefully."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.print_report()

        captured = capsys.readouterr()
        assert "No timing data available" in captured.out

    def test_print_report_with_data(self, capsys):
        """Verify print_report() generates formatted output."""
        tracker = TimingTracker()
        tracker.reset()

        # Add some timing data
        tracker.add_time('diffusion', 60.0)      # 1m
        tracker.add_time('depth_warping', 30.0)  # 30s
        tracker.add_time('video_stitching', 10.0) # 10s
        tracker.total_time = 105.0  # 1m 45s

        tracker.print_report()

        captured = capsys.readouterr()

        # Check for report structure
        assert "GENERATION TIMING REPORT" in captured.out
        assert "Total Generation Time:" in captured.out
        assert "Tracked Operations:" in captured.out
        assert "Breakdown by Operation:" in captured.out

        # Check for formatted categories
        assert "Diffusion" in captured.out
        assert "Depth Warping" in captured.out
        assert "Video Stitching" in captured.out

        # Check for bars (█ character)
        assert "█" in captured.out

        # Check for time formatting
        assert "1m" in captured.out
        assert "30s" in captured.out

    def test_format_time_seconds(self, capsys):
        """Verify time formatting for seconds only."""
        tracker = TimingTracker()
        tracker.reset()
        tracker.add_time('diffusion', 45.0)
        tracker.total_time = 45.0

        tracker.print_report()
        captured = capsys.readouterr()

        assert "45s" in captured.out

    def test_format_time_minutes(self, capsys):
        """Verify time formatting for minutes and seconds."""
        tracker = TimingTracker()
        tracker.reset()
        tracker.add_time('diffusion', 125.0)  # 2m 5s
        tracker.total_time = 125.0

        tracker.print_report()
        captured = capsys.readouterr()

        assert "2m 05s" in captured.out

    def test_format_time_hours(self, capsys):
        """Verify time formatting for hours, minutes, and seconds."""
        tracker = TimingTracker()
        tracker.reset()
        tracker.add_time('diffusion', 7325.0)  # 2h 2m 5s
        tracker.total_time = 7325.0

        tracker.print_report()
        captured = capsys.readouterr()

        assert "2h 02m 05s" in captured.out

    def test_report_sorting(self, capsys):
        """Verify operations are sorted by time (descending)."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('preprocessing', 5.0)
        tracker.add_time('diffusion', 50.0)
        tracker.add_time('depth_warping', 20.0)
        tracker.add_time('postprocessing', 3.0)
        tracker.total_time = 78.0

        tracker.print_report()
        captured = capsys.readouterr()

        # Check order by finding positions
        lines = captured.out
        diffusion_pos = lines.find('Diffusion')
        depth_warp_pos = lines.find('Depth Warping')
        preprocessing_pos = lines.find('Preprocessing')
        postprocessing_pos = lines.find('Postprocessing')

        # Verify descending order
        assert diffusion_pos < depth_warp_pos < preprocessing_pos < postprocessing_pos

    def test_untracked_overhead_display(self, capsys):
        """Verify untracked overhead is shown when significant."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('diffusion', 50.0)
        tracker.total_time = 55.0  # 5s untracked

        tracker.print_report()
        captured = capsys.readouterr()

        assert "Untracked Overhead" in captured.out

    def test_untracked_overhead_hidden(self, capsys):
        """Verify untracked overhead is hidden when insignificant."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('diffusion', 50.0)
        tracker.total_time = 50.5  # Only 0.5s untracked

        tracker.print_report()
        captured = capsys.readouterr()

        assert "Untracked Overhead" not in captured.out

    def test_percentage_calculations(self, capsys):
        """Verify percentage calculations are accurate."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('diffusion', 60.0)      # 60%
        tracker.add_time('depth_warping', 30.0)  # 30%
        tracker.add_time('video_stitching', 10.0) # 10%
        tracker.total_time = 100.0

        tracker.print_report()
        captured = capsys.readouterr()

        assert "60.0%" in captured.out
        assert "30.0%" in captured.out
        assert "10.0%" in captured.out

    def test_zero_categories_not_shown(self, capsys):
        """Verify categories with 0.0 time are not displayed."""
        tracker = TimingTracker()
        tracker.reset()

        tracker.add_time('diffusion', 50.0)
        # Leave all other categories at 0.0
        tracker.total_time = 50.0

        tracker.print_report()
        captured = capsys.readouterr()

        # Should show diffusion
        assert "Diffusion" in captured.out

        # Should NOT show categories with 0 time
        assert "Wan Interpolation" not in captured.out
        assert "Depth Warping" not in captured.out

    def test_track_exception_handling(self):
        """Verify track() context manager handles exceptions correctly."""
        tracker = TimingTracker()
        tracker.reset()

        # Exception during tracking should still accumulate time
        try:
            with tracker.track('diffusion'):
                time.sleep(0.02)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Time should still be recorded
        assert tracker.categories['diffusion'] >= 0.02

    def test_concurrent_tracking(self):
        """Verify nested tracking accumulates correctly."""
        tracker = TimingTracker()
        tracker.reset()

        with tracker.track('diffusion'):
            time.sleep(0.02)

            # Nested preprocessing within diffusion
            with tracker.track('preprocessing'):
                time.sleep(0.01)

        # Both should have accumulated time
        assert tracker.categories['diffusion'] >= 0.02
        assert tracker.categories['preprocessing'] >= 0.01

        # Diffusion should NOT include preprocessing time
        # (they're tracked separately)
        assert tracker.categories['diffusion'] < 0.04


class TestTimingTrackerIntegration:
    """Integration tests for TimingTracker in realistic scenarios."""

    def test_full_generation_cycle(self):
        """Simulate a full generation cycle."""
        tracker = get_timing_tracker()
        tracker.start_generation()

        # Simulate preprocessing
        with tracker.track('preprocessing'):
            time.sleep(0.01)

        # Simulate diffusion
        with tracker.track('diffusion'):
            time.sleep(0.03)

        # Simulate depth estimation
        with tracker.track('depth_estimation'):
            time.sleep(0.01)

        # Simulate depth warping
        with tracker.track('depth_warping'):
            time.sleep(0.02)

        # Simulate postprocessing
        with tracker.track('postprocessing'):
            time.sleep(0.01)

        tracker.end_generation()

        # Verify all categories accumulated
        assert tracker.categories['preprocessing'] >= 0.01
        assert tracker.categories['diffusion'] >= 0.03
        assert tracker.categories['depth_estimation'] >= 0.01
        assert tracker.categories['depth_warping'] >= 0.02
        assert tracker.categories['postprocessing'] >= 0.01

        # Verify total time
        assert tracker.total_time >= 0.08

    def test_multiple_generations(self):
        """Verify tracker can handle multiple generation cycles."""
        tracker = get_timing_tracker()

        # First generation
        tracker.start_generation()
        with tracker.track('diffusion'):
            time.sleep(0.02)
        tracker.end_generation()

        first_total = tracker.total_time
        first_diffusion = tracker.categories['diffusion']

        # Second generation should reset
        tracker.start_generation()
        with tracker.track('diffusion'):
            time.sleep(0.01)
        tracker.end_generation()

        second_total = tracker.total_time
        second_diffusion = tracker.categories['diffusion']

        # Second generation should have different times
        assert second_total < first_total
        assert second_diffusion < first_diffusion
