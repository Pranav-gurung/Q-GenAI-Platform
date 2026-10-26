import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from module_a.augment_images import AugmentConfig, augment_image, augment_records, rotate_bbox, _rng_for
from module_a.feedback import FeedbackStore, apply_approved
from module_a.fixtures import FIXTURE_CLASSES, generate_fixture
from module_a.ingest import load_annotations
from module_a.prepare_dataset import DatasetValidationError, prepare
from module_a.schema import Annotation, ImageRecord, bbox_problem, load_canonical
from module_a.split_dataset import check_leakage, split_records
from module_a.validate_dataset import validate_records
from module_a.version_dataset import load_manifest, verify_manifest

CLASSES = ["scratch", "dent"]


def _img(path: Path, size=(100, 100), box=None, value=255, seed=0):
    arr = np.random.default_rng(seed).integers(100, 140, (size[1], size[0]), dtype=np.uint8)
    if box:
        x0, y0, x1, y1 = box
        arr[y0:y1, x0:x1] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).convert("RGB").save(path)


def _rec(iid, box=None, cls="scratch", group=None, w=100, h=100):
    anns = [Annotation(cls, list(box))] if box else []
    return ImageRecord(image_id=iid, file_name=f"{iid}.png", width=w, height=h, group_id=group, annotations=anns)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TestIngestValidate(Base):
    def test_valid_csv_ingestion(self):
        _img(self.tmp / "a.png", box=(10, 10, 40, 30))
        (self.tmp / "a.csv").write_text("image,class,xmin,ymin,xmax,ymax\na.png,scratch,10,10,40,30\n")
        recs, issues = load_annotations(self.tmp / "a.csv")
        self.assertEqual(issues, [])
        rep = validate_records(recs, self.tmp, CLASSES)
        self.assertTrue(rep.valid, rep.errors)
        self.assertEqual((recs[0].width, recs[0].height), (100, 100))
        self.assertIsNotNone(recs[0].sha256)

    def test_json_ingestion(self):
        (self.tmp / "a.json").write_text(json.dumps([{"file_name": "a.png",
                                                       "annotations": [{"class": "dent", "bbox": [1, 2, 3, 4]}]}]))
        recs, issues = load_annotations(self.tmp / "a.json")
        self.assertEqual(recs[0].annotations[0].class_name, "dent")

    def test_unparsable_row_is_reported_not_skipped_silently(self):
        (self.tmp / "a.csv").write_text("image,class,xmin,ymin,xmax,ymax\na.png,scratch,x,10,40,30\n")
        recs, issues = load_annotations(self.tmp / "a.csv")
        self.assertTrue(any(i.code == "annotation_unparsable" for i in issues))

    def test_missing_image(self):
        rep = validate_records([_rec("ghost", (1, 1, 5, 5))], self.tmp, CLASSES)
        self.assertIn("missing_image", [i.code for i in rep.errors])

    def test_corrupted_image(self):
        (self.tmp / "bad.png").write_bytes(b"not an image at all")
        rep = validate_records([_rec("bad", (1, 1, 5, 5))], self.tmp, CLASSES)
        self.assertIn("unreadable_image", [i.code for i in rep.errors])

    def test_unknown_class(self):
        _img(self.tmp / "a.png")
        rep = validate_records([_rec("a", (1, 1, 5, 5), cls="weird")], self.tmp, CLASSES)
        self.assertIn("unknown_class", [i.code for i in rep.errors])

    def test_invalid_boxes(self):
        _img(self.tmp / "a.png")
        for box, code in [((10, 10, 5, 20), "invalid_bbox"), ((10, 10, 10, 20), "invalid_bbox"),
                          ((-3, 0, 20, 20), "bbox_out_of_bounds"), ((0, 0, 101, 20), "bbox_out_of_bounds"),
                          ((0, 0, float("nan"), 5), "invalid_bbox")]:
            rep = validate_records([_rec("a", box)], self.tmp, CLASSES)
            self.assertIn(code, [i.code for i in rep.errors], box)
        self.assertIsNotNone(bbox_problem([1, 2, 3]))

    def test_dimension_mismatch_and_empty_annotation(self):
        _img(self.tmp / "a.png")
        rep = validate_records([_rec("a", (1, 1, 5, 5), w=64, h=64)], self.tmp, CLASSES)
        self.assertIn("dimension_mismatch", [i.code for i in rep.errors])
        rep = validate_records([_rec("a")], self.tmp, CLASSES)
        self.assertIn("empty_annotation", [i.code for i in rep.warnings])
        rep = validate_records([_rec("a")], self.tmp, CLASSES, allow_empty=False)
        self.assertIn("empty_annotation", [i.code for i in rep.errors])

    def test_duplicate_and_near_duplicate_warnings(self):
        _img(self.tmp / "a.png", seed=1)
        shutil.copy(self.tmp / "a.png", self.tmp / "b.png")  # exact duplicate
        arr = np.asarray(Image.open(self.tmp / "a.png")).astype(np.int16) + 1  # near duplicate
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(self.tmp / "c.png")
        _img(self.tmp / "d.png", seed=99, box=(0, 0, 60, 100), value=0)  # clearly different
        recs = [_rec(i, (1, 1, 9, 9)) for i in "abcd"]
        rep = validate_records(recs, self.tmp, CLASSES)
        codes = [i.code for i in rep.warnings]
        self.assertIn("exact_duplicate", codes)
        self.assertIn("near_duplicate", codes)


class TestAugmentation(Base):
    def _check_tight(self, ang):
        box = (60, 20, 90, 40)
        _img(self.tmp / "a.png", box=box, value=255)
        cfg = AugmentConfig(p_rotate=1.0, rotation_max_deg=0)  # rotation applied manually below
        img = Image.open(self.tmp / "a.png").convert("L")
        rot = np.asarray(img.rotate(ang, resample=Image.BILINEAR))
        ys, xs = np.where(rot > 200)
        actual = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        pred = rotate_bbox(list(box), ang, 100, 100)
        for a, p in zip(actual, pred):
            self.assertLess(abs(a - p), 1.5)

    def test_rotation_annotation_matches_pixels(self):
        for ang in (-12, -5, 7, 15):
            self._check_tight(ang)

    def test_flip_updates_boxes_and_pixels(self):
        box = (10, 20, 30, 40)
        _img(self.tmp / "a.png", box=box)
        cfg = AugmentConfig(allow_hflip=True, p_flip=1.0, p_rotate=0, p_crop=0, p_photometric=0, p_noise=0, p_blur=0)
        rng, nprng = _rng_for(1, "a", 0)
        img, anns, ops, dropped = augment_image(Image.open(self.tmp / "a.png"), [Annotation("scratch", list(box))], cfg, rng, nprng)
        self.assertEqual(anns[0].bbox, [70.0, 20.0, 90.0, 40.0])
        self.assertEqual([o["op"] for o in ops], ["hflip"])
        arr = np.asarray(img.convert("L"))
        self.assertTrue((arr[20:40, 70:90] == 255).all())

    def test_flips_disabled_by_default(self):
        _img(self.tmp / "a.png")
        cfg = AugmentConfig(p_flip=1.0, p_rotate=0, p_crop=0)
        for k in range(5):
            rng, nprng = _rng_for(1, "a", k)
            _, _, ops, _ = augment_image(Image.open(self.tmp / "a.png"), [], cfg, rng, nprng)
            self.assertFalse(any("flip" in o["op"] for o in ops))

    def test_crop_shifts_and_clips_boxes(self):
        box = (60, 20, 90, 40)
        _img(self.tmp / "a.png", box=box)
        cfg = AugmentConfig(p_crop=1.0, crop_min_scale=0.8, p_rotate=0, p_photometric=0, p_noise=0, p_blur=0, min_visible_ratio=0.0)
        for k in range(10):
            rng, nprng = _rng_for(3, "a", k)
            img, anns, ops, _ = augment_image(Image.open(self.tmp / "a.png"), [Annotation("scratch", list(box))], cfg, rng, nprng)
            if not anns:
                continue
            b = anns[0].bbox
            arr = np.asarray(img.convert("L"))
            ys, xs = np.where(arr > 200)
            self.assertAlmostEqual(b[0], xs.min(), delta=1.1)
            self.assertAlmostEqual(b[2], xs.max() + 1, delta=1.1)
            self.assertLessEqual(b[2], img.size[0])

    def test_reproducible_with_seed(self):
        _img(self.tmp / "a.png", box=(10, 10, 30, 30))
        r = _rec("a", (10, 10, 30, 30)); r.split = "train"
        cfg = AugmentConfig(seed=5, variants_per_image=3)
        n1, _ = augment_records([r], self.tmp, self.tmp / "o1", cfg)
        n2, _ = augment_records([r], self.tmp, self.tmp / "o2", cfg)
        self.assertEqual([x.annotations[0].bbox for x in n1], [x.annotations[0].bbox for x in n2])
        self.assertEqual((self.tmp / "o1" / n1[0].file_name).read_bytes(), (self.tmp / "o2" / n2[0].file_name).read_bytes())
        n3, _ = augment_records([r], self.tmp, self.tmp / "o3", AugmentConfig(seed=6, variants_per_image=3))
        self.assertNotEqual([x.aug_ops for x in n1], [x.aug_ops for x in n3])

    def test_refuses_non_train_records(self):
        _img(self.tmp / "a.png")
        r = _rec("a"); r.split = "val"
        with self.assertRaises(ValueError):
            augment_records([r], self.tmp, self.tmp / "o", AugmentConfig())

    def test_mask_only_annotation_rejected(self):
        _img(self.tmp / "a.png")
        with self.assertRaises(NotImplementedError):
            rng, nprng = _rng_for(1, "a", 0)
            augment_image(Image.open(self.tmp / "a.png"), [Annotation("scratch", None, "m.png")], AugmentConfig(), rng, nprng)


class TestSplit(Base):
    def _records(self, n_groups=30, per_group=2):
        recs = []
        for g in range(n_groups):
            for v in range(per_group):
                r = _rec(f"g{g}_v{v}", (5, 5, 20, 20), cls=CLASSES[g % 2], group=f"g{g}")
                r.sha256, r.phash = f"h{g}_{v}", f"{(g * 2654435761 + v * 17) & 0xFFFFFFFFFFFFFFFF:016x}"
                recs.append(r)
        return recs

    def test_reproducible_and_seed_sensitive(self):
        a, b, c = self._records(), self._records(), self._records()
        split_records(a, seed=1); split_records(b, seed=1); split_records(c, seed=2)
        self.assertEqual({r.image_id: r.split for r in a}, {r.image_id: r.split for r in b})
        self.assertNotEqual({r.image_id: r.split for r in a}, {r.image_id: r.split for r in c})

    def test_groups_never_split_and_ratios_approx(self):
        recs = self._records()
        stats, issues = split_records(recs, seed=3)
        groups = {}
        for r in recs:
            groups.setdefault(r.group_id, set()).add(r.split)
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        self.assertEqual(check_leakage(recs), [])
        self.assertAlmostEqual(stats["splits"]["train"]["ratio_actual"], 0.7, delta=0.1)

    def test_exact_duplicates_across_groups_stay_together(self):
        recs = self._records(n_groups=20)
        recs[0].sha256 = recs[-1].sha256 = "SAME"
        split_records(recs, seed=1)
        self.assertEqual(recs[0].split, recs[-1].split)

    def test_near_duplicates_stay_together(self):
        recs = self._records(n_groups=20)
        recs[-1].phash = recs[0].phash  # distance 0, different group
        split_records(recs, seed=1)
        self.assertEqual(recs[0].split, recs[-1].split)

    def test_leakage_check_detects_problems(self):
        recs = self._records(n_groups=6)
        split_records(recs, seed=1)
        a = next(r for r in recs if r.split == "train")
        b = next(r for r in recs if r.split != "train")
        b.group_id = a.group_id  # simulate leakage
        self.assertIn("group_leak", [i.code for i in check_leakage(recs)])

    def test_warnings_small_and_missing_class(self):
        recs = [_rec(f"i{i}", (1, 1, 5, 5), cls="scratch", group=f"g{i}") for i in range(10)]
        for r in recs:
            r.sha256, r.phash = r.image_id, f"{hash(r.image_id) & 0xFFFFFFFFFFFFFFFF:016x}"
        recs.append(_rec("d", (1, 1, 5, 5), cls="dent", group="gd")); recs[-1].sha256 = "d"; recs[-1].phash = "f" * 16
        _, issues = split_records(recs, seed=1)
        codes = [i.code for i in issues]
        self.assertIn("small_eval_split", codes)
        self.assertIn("class_missing_in_split", codes)

    def test_bad_ratios(self):
        with self.assertRaises(ValueError):
            split_records(self._records(), ratios=(0.5, 0.5, 0.5))


class TestPipeline(Base):
    def test_end_to_end_fixture(self):
        img_dir, ann = generate_fixture(self.tmp / "fx", seed=1)
        res = prepare(image_dir=img_dir, annotations=ann, classes=FIXTURE_CLASSES, out_root=self.tmp / "out",
                      source={"name": "fx", "path": str(img_dir), "is_test_fixture": True, "license_notes": "n/a"},
                      export_formats=("coco", "yolo", "anomaly"), seed=7,
                      augment=AugmentConfig(seed=7, variants_per_image=2), readme_path=self.tmp / "README.md")
        self.assertGreater(res["n_augmented"], 0)
        m = load_manifest(res["manifest"])
        self.assertEqual(m["random_seed"], 7)
        self.assertEqual(verify_manifest(m, res["layout"]["splits_dir"]), [])
        for key in ("source", "class_map", "split_assignments", "augmentation", "validation", "known_limitations", "created_at"):
            self.assertIn(key, m)
        # augmented images only in train; val/test contain only originals
        for s in ("val", "test"):
            recs, _, _ = load_canonical(Path(res["layout"]["annotations_dir"]) / f"canonical_{s}.json")
            self.assertTrue(all(r.source_image_id is None for r in recs))
        train, _, _ = load_canonical(Path(res["layout"]["annotations_dir"]) / "canonical_train.json")
        self.assertTrue(any(r.source_image_id for r in train))
        # every augmented box lies inside its image
        for r in train:
            for a in r.annotations:
                self.assertIsNone(bbox_problem(a.bbox, r.width, r.height), (r.image_id, a.bbox))
        # original dataset untouched
        self.assertEqual(len(list(img_dir.glob("*.png"))), 14 * 3)
        self.assertIn("TEST FIXTURE", (self.tmp / "README.md").read_text())
        # coco/yolo exist
        self.assertTrue((Path(res["layout"]["processed_dir"]) / "coco" / "instances_train.json").exists())
        self.assertTrue((Path(res["layout"]["processed_dir"]) / "yolo" / "data.yaml").exists())

    def test_same_inputs_same_version_and_splits(self):
        img_dir, ann = generate_fixture(self.tmp / "fx", seed=1)
        kw = dict(image_dir=img_dir, annotations=ann, classes=FIXTURE_CLASSES, source={"name": "fx", "path": "x"}, seed=3,
                  augment=AugmentConfig(seed=3, variants_per_image=1))
        r1 = prepare(out_root=self.tmp / "o1", **kw)
        r2 = prepare(out_root=self.tmp / "o2", **kw)
        self.assertEqual(r1["version"], r2["version"])
        m1, m2 = load_manifest(r1["manifest"]), load_manifest(r2["manifest"])
        self.assertEqual(m1["split_assignments"], m2["split_assignments"])

    def test_invalid_dataset_fails_loudly_and_quarantine_reports(self):
        img_dir, ann = generate_fixture(self.tmp / "fx", seed=1)
        (img_dir / "part000_view0.png").write_bytes(b"corrupt")
        kw = dict(image_dir=img_dir, annotations=ann, classes=FIXTURE_CLASSES, source={"name": "fx", "path": "x"},
                  augment=AugmentConfig(variants_per_image=0))
        with self.assertRaises(DatasetValidationError):
            prepare(out_root=self.tmp / "o", **kw)
        res = prepare(out_root=self.tmp / "o", quarantine_invalid=True, **kw)
        rep = json.loads((Path(res["layout"]["reports_dir"]) / "validation.json").read_text())
        self.assertIn("part000_view0", rep["invalid_image_ids"])


class TestFeedback(Base):
    def setUp(self):
        super().setUp()
        self.store = FeedbackStore(self.tmp / "fb.jsonl", CLASSES)

    def _submit(self, **kw):
        base = dict(image_id="a", dataset_version="ds-1", original_class="scratch", corrected_class="dent", submitted_by="op1")
        base.update(kw)
        return self.store.submit(**base)

    def test_pending_not_exported_or_applied(self):
        fb = self._submit()
        self.assertEqual(self.store.export_approved(self.tmp / "ap.json"), 0)
        recs, log = apply_approved([_rec("a", (1, 1, 9, 9))], [fb])
        self.assertEqual(recs[0].annotations[0].class_name, "scratch")
        self.assertFalse(log[0]["applied"])

    def test_approval_flow_and_four_eyes(self):
        fb = self._submit()
        with self.assertRaises(ValueError):
            self.store.review(fb.feedback_id, "approved", "op1")  # self-approval
        self.store.review(fb.feedback_id, "approved", "lead", "ok")
        with self.assertRaises(ValueError):
            self.store.review(fb.feedback_id, "rejected", "lead")  # already decided
        self.assertEqual(self.store.export_approved(self.tmp / "ap.json", "ds-1"), 1)
        orig = [_rec("a", (1, 1, 9, 9))]
        new, log = apply_approved(orig, self.store.list("approved"))
        self.assertEqual(new[0].annotations[0].class_name, "dent")
        self.assertEqual(orig[0].annotations[0].class_name, "scratch")  # input not mutated
        self.assertTrue(log[0]["applied"])

    def test_rejected_not_applied(self):
        fb = self._submit()
        self.store.review(fb.feedback_id, "rejected", "lead")
        _, log = apply_approved([_rec("a", (1, 1, 9, 9))], self.store.list())
        self.assertFalse(log[0]["applied"])

    def test_validation_on_submit(self):
        with self.assertRaises(ValueError):
            self._submit(corrected_class="bogus")
        with self.assertRaises(ValueError):
            self._submit(corrected_class=None, corrected_bbox=[5, 5, 1, 1])
        with self.assertRaises(ValueError):
            self._submit(corrected_class=None)
        with self.assertRaises(ValueError):
            self._submit(corrected_class=None, corrected_bbox=[0, 0, 200, 50], image_size=(100, 100))

    def test_ambiguous_target_not_guessed(self):
        fb = self._submit()
        self.store.review(fb.feedback_id, "approved", "lead")
        rec = _rec("a", (1, 1, 9, 9)); rec.annotations.append(Annotation("scratch", [20, 20, 30, 30]))
        _, log = apply_approved([rec], self.store.list("approved"))
        self.assertFalse(log[0]["applied"])

    def test_apply_approved_from_dict_loaded_from_json(self):
        fb = self._submit()
        self.store.review(fb.feedback_id, "approved", "lead")
        out_json = self.tmp / "approved.json"
        self.store.export_approved(out_json)
        loaded_dicts = json.loads(out_json.read_text(encoding="utf-8"))
        orig = [_rec("a", (1, 1, 9, 9))]
        new, log = apply_approved(orig, loaded_dicts)
        self.assertEqual(new[0].annotations[0].class_name, "dent")
        self.assertTrue(log[0]["applied"])


if __name__ == "__main__":
    unittest.main()
