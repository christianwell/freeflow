import io
import json
import signal
import socket
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import freeflow


class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.wav"
        with wave.open(str(self.path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00\x02\x00" * 80)

    def app(self, returncode=1, running=True, stderr=b""):
        app = freeflow.FreeFlowLinux.__new__(freeflow.FreeFlowLinux)
        app.audio_path = self.path
        app.config = dict(freeflow.DEFAULTS)
        app.recording = Mock(returncode=returncode)
        app.recording.poll.return_value = None if running else returncode
        app.recording.communicate.return_value = (b"", stderr)
        app.record_button = Mock()
        app.status = Mock()
        app.show_error = Mock()
        app.busy = False
        return app

    def test_normal_alsa_sigint_exit_one_is_processed(self):
        app = self.app()
        process = app.recording
        with patch.object(freeflow.threading, "Thread") as worker:
            app.stop_recording()
        process.send_signal.assert_called_once_with(signal.SIGINT)
        worker.return_value.start.assert_called_once()
        app.show_error.assert_not_called()
        self.assertTrue(app.busy)
        self.assertTrue(self.path.exists())

    def test_recorder_failures_never_reach_provider(self):
        for running, code, stderr in [(False, 1, b""), (True, 1, b"device unavailable"), (True, 2, b"")]:
            with self.subTest(running=running, code=code, stderr=stderr):
                app = self.app(code, running, stderr)
                with patch.object(freeflow.threading, "Thread") as worker:
                    app.stop_recording()
                worker.assert_not_called()
                app.show_error.assert_called_once()
                self.assertFalse(self.path.exists())
                self.assertIsNone(app.audio_path)

    def test_timeout_is_not_accepted_as_normal_stop(self):
        app = self.app()
        process = app.recording
        process.communicate.side_effect = [subprocess.TimeoutExpired("arecord", 5), (b"", b"")]
        with patch.object(freeflow.threading, "Thread") as worker:
            app.stop_recording()
        process.kill.assert_called_once()
        worker.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_audio_validation_rejects_empty_corrupt_and_truncated_wav(self):
        valid = self.path.read_bytes()
        freeflow.validate_audio(self.path)
        for data in [b"", b"not a WAV" * 10, valid[:44], valid[:-2]]:
            with self.subTest(size=len(data)):
                self.path.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError, "No valid microphone audio"):
                    freeflow.validate_audio(self.path)

    def test_invalid_audio_is_deleted_on_stop(self):
        self.path.write_bytes(b"broken" * 20)
        app = self.app()
        with patch.object(freeflow.threading, "Thread") as worker:
            app.stop_recording()
        worker.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_wrong_audio_format_is_rejected(self):
        with wave.open(str(self.path), "wb") as audio:
            audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00\x02\x00" * 80)
        with self.assertRaisesRegex(RuntimeError, "No valid microphone audio"):
            freeflow.validate_audio(self.path)

    def test_transcription_request_uses_separate_provider(self):
        config = dict(freeflow.DEFAULTS, api_key="synthetic-cleanup-key",
                      transcription_api_key="synthetic-transcription-key",
                      transcription_api_url="https://transcription.example/v1/", language="da")
        with patch.object(freeflow, "request_json", return_value={"text": "  Synthetic words.  "}) as request:
            self.assertEqual(freeflow.transcribe(config, self.path), "Synthetic words.")
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, "https://transcription.example/v1/audio/transcriptions")
        self.assertEqual(sent.get_header("Authorization"), "Bearer synthetic-transcription-key")
        self.assertIn(b'\r\n\r\nda\r\n', sent.data)
        self.assertIn(self.path.read_bytes(), sent.data)

    def test_config_file_has_private_permissions(self):
        folder = Path(self.directory.name)
        config_path = folder / "config.json"
        config_path.write_text("{}")
        config_path.chmod(0o644)
        with patch.object(freeflow, "APP_DIR", folder), patch.object(freeflow, "CONFIG_PATH", config_path):
            freeflow.save_config(dict(freeflow.DEFAULTS, api_key="synthetic-key"))
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(freeflow.load_config()["api_key"], "synthetic-key")

    def test_failed_save_preserves_original_and_removes_temporary_file(self):
        folder = Path(self.directory.name)
        config_path = folder / "config.json"
        config_path.write_text('{"api_key":"synthetic-original-key"}')
        with patch.object(freeflow, "APP_DIR", folder), patch.object(freeflow, "CONFIG_PATH", config_path), \
                patch.object(freeflow.os, "replace", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                freeflow.save_config(dict(freeflow.DEFAULTS, api_key="synthetic-new-key"))
        self.assertEqual(json.loads(config_path.read_text())["api_key"], "synthetic-original-key")
        self.assertEqual(list(folder.glob(".config-*")), [])

    def test_invalid_config_types_use_defaults(self):
        config_path = Path(self.directory.name) / "config.json"
        for saved in [[], "api_key", {"api_key": 123, "cleanup": "false", "language": "da"}]:
            with self.subTest(saved=saved):
                config_path.write_text(json.dumps(saved))
                with patch.object(freeflow, "CONFIG_PATH", config_path):
                    config = freeflow.load_config()
                self.assertEqual(config["api_key"], "")
                self.assertIs(config["cleanup"], True)
                self.assertEqual(config["language"], "da" if isinstance(saved, dict) else "")

    def test_failed_settings_save_prevents_recording(self):
        app = self.app()
        app.current_config = Mock(return_value=dict(freeflow.DEFAULTS))
        with patch.object(freeflow, "save_config", side_effect=OSError()), \
                patch.object(freeflow.shutil, "which", return_value="arecord"), \
                patch.object(freeflow.subprocess, "Popen") as recorder:
            app.start_recording()
        recorder.assert_not_called()
        app.show_error.assert_called_once()

    def test_provider_redirect_never_receives_credentials_or_content(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(302)
                self.send_header("Location", "/destination")
                self.end_headers()

            def do_GET(self):
                received.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/source",
                                             data=b"synthetic content",
                                             headers={"Authorization": "Bearer synthetic-key"})
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                freeflow.request_json(request)
            self.assertEqual(received, ["/source"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_provider_error_body_is_not_displayed(self):
        error = urllib.error.HTTPError("https://provider.example", 401, "Unauthorized", {},
                                       io.BytesIO(b'{"error":{"message":"synthetic-private-content"}}'))
        opener = Mock()
        opener.open.side_effect = error
        with patch.object(freeflow.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(RuntimeError) as raised:
                freeflow.request_json(urllib.request.Request("https://provider.example"))
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("synthetic-private-content", str(raised.exception))

    def test_provider_invalid_json_shapes_are_reported(self):
        for body in [b"[]", b"null", b"not JSON", b"\xff"]:
            with self.subTest(body=body):
                response = Mock()
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.read.return_value = body
                opener = Mock()
                opener.open.return_value = response
                with patch.object(freeflow.urllib.request, "build_opener", return_value=opener):
                    with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                        freeflow.request_json(urllib.request.Request("https://provider.example"))

    def test_service_readiness_is_sent_after_app_initialization(self):
        address = str(Path(self.directory.name) / "notify.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
            receiver.bind(address)
            receiver.setblocking(False)
            root = Mock()

            def initialized(_root):
                # No premature readiness datagram while initialization is running.
                with self.assertRaises(BlockingIOError):
                    receiver.recv(64)

            with patch.dict(freeflow.os.environ, {"NOTIFY_SOCKET": address}), \
                    patch.object(freeflow.tk, "Tk", return_value=root), \
                    patch.object(freeflow, "FreeFlowLinux", side_effect=initialized):
                freeflow.main()
            receiver.settimeout(1)
            self.assertEqual(receiver.recv(64), b"READY=1")
            root.mainloop.assert_called_once()

    def test_show_signal_request_is_consumed_on_ui_thread(self):
        app = self.app()
        app.root = Mock()
        app.show_window = Mock()
        app.show_requested = True
        app._poll_show_request()
        app.show_window.assert_called_once()
        self.assertFalse(app.show_requested)
        app.root.after.assert_called_once_with(100, app._poll_show_request)
        app._poll_show_request()
        app.show_window.assert_called_once()

    def test_pipeline_cleans_pastes_and_removes_audio(self):
        app = self.app()
        app.root = Mock()
        with patch.object(freeflow, "transcribe", return_value="invented words") as transcribe, \
                patch.object(freeflow, "clean_transcript", return_value="Invented words."):
            app._process_audio(app.config, self.path)
        transcribe.assert_called_once_with(app.config, self.path)
        self.assertEqual(app.root.after.call_args.args, (0, app.paste_text, "Invented words."))
        self.assertFalse(self.path.exists())

    def test_provider_failure_deletes_audio_and_does_not_paste(self):
        app = self.app()
        app.root = Mock()
        with patch.object(freeflow, "transcribe", side_effect=RuntimeError("Synthetic provider failure")):
            app._process_audio(app.config, self.path)
        self.assertEqual(app.root.after.call_args.args, (0, app.show_error, "Synthetic provider failure"))
        self.assertFalse(self.path.exists())

    def test_clipboard_is_not_restored_over_new_user_content(self):
        app = self.app()
        app.root = Mock()
        app.root.clipboard_get.return_value = "New synthetic content"
        app._restore_clipboard("Old synthetic content", "Synthetic dictation")
        app.root.clipboard_clear.assert_not_called()

    def test_paste_failure_resets_busy_and_schedules_clipboard_restoration(self):
        app = self.app()
        del app.show_error
        app.busy = True
        app.root = Mock()
        app.root.clipboard_get.return_value = "Old synthetic clipboard"
        app.variables = {"preserve_clipboard": Mock()}
        app.variables["preserve_clipboard"].get.return_value = True
        app.backend = Mock()
        app.backend.paste.side_effect = RuntimeError("Synthetic X11 failure")
        app.notify = Mock()
        with patch.object(freeflow.messagebox, "showerror"):
            app.paste_text("Synthetic transcription")
        self.assertFalse(app.busy)
        app.root.after.assert_called_once_with(1000, app._restore_clipboard,
                                                "Old synthetic clipboard", "Synthetic transcription")
        app.root.update.assert_not_called()

    def test_notifications_do_not_include_provider_details(self):
        app = self.app()
        app.notify = Mock()
        with patch.object(freeflow.messagebox, "showerror"):
            freeflow.FreeFlowLinux.show_error(app, "Synthetic private provider detail")
        app.notify.assert_called_once_with("FreeFlow error", "Open FreeFlow for details")


if __name__ == "__main__":
    unittest.main()
