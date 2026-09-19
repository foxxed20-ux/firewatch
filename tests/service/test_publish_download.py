import hashlib

import pytest

from deploy.publish_download import publish_archive


def test_publish_verified_file_and_idempotent_replay(tmp_path):
    source = tmp_path / "incoming.zip"
    source.write_bytes(b"verified-release")
    public = tmp_path / "downloads"
    public.mkdir()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    first = publish_archive(source, public, "release-v2.zip", digest)
    assert first["status"] == "published"
    assert (public / "release-v2.zip").read_bytes() == source.read_bytes()
    assert (
        publish_archive(source, public, "release-v2.zip", digest)["status"]
        == "already_published"
    )
    assert not list(tmp_path.glob(".firewatch-download-*"))


def test_bad_hash_never_becomes_public(tmp_path):
    source = tmp_path / "incoming.zip"
    source.write_bytes(b"wrong-content")
    public = tmp_path / "downloads"
    public.mkdir()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        publish_archive(source, public, "release-v2.zip", "0" * 64)
    assert list(public.iterdir()) == []
    assert not list(tmp_path.glob(".firewatch-download-*"))


def test_published_name_cannot_be_replaced(tmp_path):
    public = tmp_path / "downloads"
    public.mkdir()
    target = public / "release-v2.zip"
    target.write_bytes(b"previous-release")
    source = tmp_path / "incoming.zip"
    source.write_bytes(b"replacement")
    with pytest.raises(FileExistsError):
        publish_archive(
            source, public, target.name, hashlib.sha256(source.read_bytes()).hexdigest()
        )
    assert target.read_bytes() == b"previous-release"
