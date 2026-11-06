from app import coco


def test_coco_image_keys_and_urls_are_stable():
    key = coco.image_key("val2017", 42)

    assert key == "val2017:42"
    assert coco.image_url(key) == "/coco/images/val2017:42/file"


def test_coco_status_defaults_to_not_imported():
    class Collection:
        def find_one(self, _query):
            return None

        def find(self, *_args, **_kwargs):
            return Cursor()

    class Cursor(list):
        def sort(self, *_args, **_kwargs):
            return self

    class Database:
        coco_imports = Collection()
        coco_categories = Collection()

    assert coco.status(Database()) == {"status": "not_imported", "splits": {}, "categories": [], "error": None}
