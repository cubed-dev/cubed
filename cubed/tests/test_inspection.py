import cubed.array_api as xp

info = xp.__array_namespace_info__()


def test_capabilities():
    capabilities = info.capabilities()
    assert capabilities["boolean indexing"] is False
    assert capabilities["data-dependent shapes"] is False


def test_default_device():
    assert (
        info.default_device() is None or info.default_device() == xp.asarray(0).device
    )


def test_default_dtypes():
    dtypes = info.default_dtypes()
    assert dtypes["real floating"] == xp.asarray(0.0).dtype
    assert dtypes["complex floating"] == xp.asarray(0.0j).dtype
    assert dtypes["integral"] == xp.asarray(0).dtype
    assert dtypes["indexing"] == xp.argmax(xp.zeros(10)).dtype


def test_devices():
    assert len(info.devices()) > 0


def test_dtypes():
    assert len(info.dtypes()) > 0
