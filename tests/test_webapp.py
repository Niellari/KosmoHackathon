"""Проверки запуска локального веб-сервиса."""

from __future__ import annotations

import errno
import unittest
from unittest.mock import MagicMock, patch

from src.webapp import find_available_port, open_in_browser


class ServerPortTests(unittest.TestCase):
    def test_busy_preferred_port_falls_back_to_next_port(self):
        probe = MagicMock()
        probe.__enter__.return_value.bind.side_effect = [
            OSError(errno.EADDRINUSE, "Address already in use"),
            None,
        ]

        with patch("src.webapp.socket.socket", return_value=probe):
            selected = find_available_port("127.0.0.1", 8000)

        self.assertEqual(selected, 8001)
        self.assertEqual(probe.__enter__.return_value.bind.call_count, 2)

    @patch("src.webapp.webbrowser.open", return_value=True)
    def test_browser_is_opened_in_new_tab(self, browser_open):
        open_in_browser("http://127.0.0.1:8001/")

        browser_open.assert_called_once_with("http://127.0.0.1:8001/", new=2)


if __name__ == "__main__":
    unittest.main()
