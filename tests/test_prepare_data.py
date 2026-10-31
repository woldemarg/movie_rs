from scripts.prepare_data import prepare


def test_prepare_builds_only_a_missing_catalogue_and_refreshes_posters_only_with_a_key(tmp_path):
    calls: list[str] = []

    def build() -> None:
        calls.append("build")
        (tmp_path / "movies.json").write_text("{}", encoding="utf-8")
        (tmp_path / "movie_vectors.npy").write_bytes(b"")

    assert "built" in prepare(tmp_path, build, lambda: calls.append("posters"), tmdb_key=False)
    assert calls == ["build"]

    assert prepare(tmp_path, build, lambda: calls.append("posters"), tmdb_key=True) == "ready"
    assert calls == ["build"]

    (tmp_path / "movie_vectors.npy").unlink()  # half a catalogue is rebuilt as a whole
    prepare(tmp_path, build, lambda: calls.append("posters"), tmdb_key=True)
    assert calls == ["build", "build", "posters"]
