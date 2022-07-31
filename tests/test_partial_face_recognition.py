#!/usr/bin/env python

"""Tests for `partial_face_recognition` package."""


import unittest
from click.testing import CliRunner

from partial_face_recognition import partial_face_recognition
from partial_face_recognition import cli


class TestPartial_face_recognition(unittest.TestCase):
    """Tests for `partial_face_recognition` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'partial_face_recognition.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
