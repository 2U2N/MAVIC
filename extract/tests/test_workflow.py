import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from mavic_extract import Project
from mavic_extract.export import export
from mavic_extract.speech import eligible
from mavic_extract.linguistic import pipeline_language

class WorkflowTests(unittest.TestCase):
    def test_gate_and_alias(self):
        self.assertFalse(eligible(.98))
        self.assertTrue(eligible(.98001))
        self.assertEqual(pipeline_language("no"), "nb")

    def test_restart_and_partial_exports(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp)/"videos"
            media.mkdir()
            (media/"short.mp4").write_bytes(b"fixture")
            with patch("mavic_extract.project.probe", return_value=(4, True)):
                p = Project.inspect(media,Path(tmp)/"results")
            with patch.object(Project,"_compute",return_value=({"avg_shots_per_min":0},{"observations":0})) as run:
                p.run("motion")
                resumed = Project(p.folder)
                self.assertEqual(resumed.run("motion").status[0],"cached")
                self.assertEqual(run.call_count,1)
            export(resumed)
            folder = p.folder/"exports"
            candidates = pd.read_csv(folder/"candidates.csv")
            self.assertEqual(candidates.shape,(1,272))
            self.assertEqual(candidates.avg_shots_per_min[0],0)
            self.assertEqual(pd.read_csv(folder/"full_features.csv").shape,(0,100))
            self.assertEqual(pd.read_csv(folder/"linguistic_features.csv").shape,(0,134))

    def test_chronological_keyframes_and_flow_pairs(self):
        import subprocess
        from mavic_extract import appearance, motion
        with TemporaryDirectory() as tmp:
            clip = Path(tmp)/"clock.mp4"
            subprocess.run(["ffmpeg","-loglevel","error","-f","lavfi","-i","testsrc2=size=64x64:rate=2","-t","120","-pix_fmt","yuv420p",str(clip)],check=True)
            def inspect_order(folder):
                indices = [int(p.stem.split("_")[-1]) for p in sorted(Path(folder).glob("*.png"))]
                self.assertEqual(indices,list(range(12)))
                return {},{}
            with patch.object(appearance,"summarize",inspect_order):
                _,detail = appearance.measure(clip,120,Path(tmp)/"preview.png")
            self.assertEqual(detail["timestamps"],list(range(5,120,10)))
            _,detail = motion.process_video(clip)
            self.assertEqual(detail["frame_pairs"][:3],[[0,10],[10,30],[30,50]])

    def test_resume_after_interruption(self):
        with TemporaryDirectory() as tmp:
            videos = Path(tmp)/"videos"
            videos.mkdir()
            for name in ("a.mp4","b.mp4"):
                (videos/name).write_bytes(name.encode())
            with patch("mavic_extract.project.probe", return_value=(20,True)):
                p = Project.inspect(videos,Path(tmp)/"results")
            result = ({"mean_freq_centroid": 440},{"observations":2})
            with patch.object(Project,"_compute",side_effect=[result,KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    p.run("acoustic")
            with patch.object(Project,"_compute",return_value=result) as call:
                statuses = Project(p.folder).run("acoustic").status.tolist()
                self.assertEqual(statuses,["cached","completed"])
                self.assertEqual(call.call_count,1)
            (videos/"a.mp4").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"changed"):
                p.run("acoustic")

    def test_unsupported_language(self):
        from mavic_extract.linguistic import measure
        values,detail = measure({"detail":{"language":"ar","text":"مرحبا"}},5,["en_core_web_md"])
        self.assertEqual(values,{})
        self.assertIn("No selected pipeline",detail["reason"])
