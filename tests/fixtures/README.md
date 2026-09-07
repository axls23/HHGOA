# Test fixtures

Two images, both of the same subject, both covered by the same consent as any
other run of this pipeline (Lane A, `docs/CONSENT.md`): they are frames of the
repository author's own face, captured from their own webcam by
`scripts/capture_probe.py` and cropped for use here.

That is recorded explicitly because a project about consent for face
recognition has no business shipping a stranger's face in its test suite, and
because "where did this face come from" is a question a reader should never
have to guess at. Anyone forking this repo should replace both with faces they
are entitled to use.

| File | Size | What it is for |
|---|---|---|
| `probe_sample.jpg` | 112×112 | The probe fixture. Cropped so the detected bbox is **78.7 px** — deliberately just under the gate's 80 px floor, so `test_quality_gate_rejects_fixture_min_bbox` documents that exact boundary and the other tests upscale 2× to get a clean pass. |
| `capture_frame.jpg` | 480×480 | A camera *frame*, with the context around the face that a 112×112 crop does not have. Used by the liveness and frame-selection tests: MiniFASNet widens the detection box 2.7–4× before it looks, so on a tight crop it sees none of what it was trained on and calls a real face an attack. |
| `google_web_detection.json` | — | A recorded Google Cloud Vision `images:annotate` response, replayed by `--provider mock`. Contains no image data; URLs point at public pages. |

Regenerate the two images from a probe of your own with:

```bash
python scripts/capture_probe.py --out evidence/probe.jpg    # your face, your consent
# then crop: 1.45× the detected bbox → 112×112, and 2.6× → 480×480
```

The exact crop factors matter — see the table above for what each test reads
off them.
