from util.major_mapping import major_mapping


def test_major_mapping_normalization():
    assert isinstance(major_mapping, dict)
    assert len(major_mapping) > 0
